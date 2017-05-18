package com.sample.spark.task;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import scala.Tuple2;

import java.io.File;
import java.io.FileOutputStream;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Created by chandrashekar.v on 5/8/2017.
 */
public class RecommendationEngineV2 {

    private static final boolean ENABLE_DEBUG_MODE = false;
    private static final int R = 5;
    private static final int C1 = 5;
    private static final int C2 = 5;
    public static final String path = "log.txt";
    public static final String contentInfoPath = "content.csv";

    private static final Function2<Double, Double, Double> RATING_FUNCTION = (recency, frequency) -> (C1 * recency) + (C2 * frequency);

    public static void main(String[] args) {
        generateLogEntries();

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        SparkSession sparkSession = SparkSession.builder().appName("RecommendationEngine").config("spark.master", "local").getOrCreate();
        JavaSparkContext sc = new JavaSparkContext(sparkSession.sparkContext());


        JavaRDD<String> data = sparkSession.read().textFile(path).javaRDD();
        JavaPairRDD<String, LogEntry> sessionToLogEntry = data.mapToPair(e -> {
            String[] tokens = e.split(",");
            LogEntry entry = new LogEntry();
            entry.setSessionId(tokens[0]);
            entry.setUserId(tokens[1]);
            entry.setContentId(tokens[2]);
            entry.setTime(Long.valueOf(tokens[3]));
            return new Tuple2<String, LogEntry>(tokens[0], entry);
        });

        JavaPairRDD<String, Double> userContentRecencyRDD = data.mapToPair(e -> {
            String[] tokens = e.split(",");
            long timeElapsed = System.currentTimeMillis() - Long.valueOf(tokens[3]);
            return new Tuple2<String, Long>(tokens[1] + "_" + tokens[2], timeElapsed);
        }).reduceByKey((aLong, aLong2) -> Math.min(aLong, aLong2))
                .mapToPair(t -> new Tuple2<String, Double>(t._1, Math.exp(-R * TimeUnit.MILLISECONDS.toDays(t._2()))));

        printAfterRecencyCalculation(userContentRecencyRDD);

        JavaPairRDD<String, LogEntry> sessionLogEntryJavaPairRDD = sessionToLogEntry.reduceByKey(new Function2<LogEntry, LogEntry, LogEntry>() {
            @Override
            public LogEntry call(LogEntry logEntry1, LogEntry logEntry2) throws Exception {
                LogEntry result = SerializationUtils.clone(logEntry1);
                result.setTime(getTimeDiff(logEntry1.getTime(), logEntry2.getTime()));
                return result;
            }
        });

        printAfterTimeDiffCalculation(sessionLogEntryJavaPairRDD);

        JavaPairRDD<String, LogEntry> userContentKeyLogEntryPairRDD = sessionLogEntryJavaPairRDD.mapToPair(new PairFunction<Tuple2<String, LogEntry>, String, LogEntry>() {
            @Override
            public Tuple2<String, LogEntry> call(Tuple2<String, LogEntry> logEntryTuple) throws Exception {
                final String key = logEntryTuple._2().getUserId() + "_" + logEntryTuple._2().getContentId();
                return new Tuple2<String, LogEntry>(key, logEntryTuple._2());
            }
        });

        printAfterMappingUserAndContentIdToLogEntry(userContentKeyLogEntryPairRDD);

        JavaPairRDD<String, LogEntry> totalTimeLogEntryPairRDD = userContentKeyLogEntryPairRDD.reduceByKey(new Function2<LogEntry, LogEntry, LogEntry>() {
            @Override
            public LogEntry call(LogEntry logEntry, LogEntry logEntry2) throws Exception {

                long totalTime = logEntry.getTime() + logEntry2.getTime();
                LogEntry logEntry1 = new LogEntry();
                logEntry1.setTime(totalTime);
                return logEntry1;
            }
        });

        // JavaPairRDD<String, Long> userContentTotalTimeJavaPairRDD = totalTimeLogEntryPairRDD.mapToPair(t -> new Tuple2<>(t._1(), t._2().getTime()));

        // Join Recency And Total Time RDD into single Pair.
        JavaPairRDD<String, Tuple2<Double, LogEntry>> recencyTotalTimeJoinRDD = userContentRecencyRDD
                .join(totalTimeLogEntryPairRDD);
        JavaPairRDD<String, LogEntry> recencyTotalTimeJoinRDDPair = recencyTotalTimeJoinRDD.mapToPair(tuple -> {
            tuple._2()._2().setRecency(tuple._2()._1());
            return new Tuple2<String, LogEntry>(tuple._1(), tuple._2()._2());
        });


        printAfterCalculatingTotalTime(totalTimeLogEntryPairRDD);

        // To find maximum difference
        JavaPairRDD<String, LogEntry> userLogEntryJavaPairRDD = totalTimeLogEntryPairRDD.mapToPair(new PairFunction<Tuple2<String, LogEntry>, String, LogEntry>() {
            @Override
            public Tuple2<String, LogEntry> call(Tuple2<String, LogEntry> logEntryTuple) throws Exception {
                return new Tuple2<String, LogEntry>(logEntryTuple._1().substring(0, logEntryTuple._1().indexOf("_")), logEntryTuple._2());
            }
        });

        printAfterMappingUserWithViewingTime(userLogEntryJavaPairRDD);

        JavaPairRDD<String, Long> userMaxTimeLogEntryJavaPairRDD = userLogEntryJavaPairRDD.reduceByKey((logEntry1, logEntry2) -> {
                    if (Math.max(logEntry1.getTime(), logEntry2.getTime()) == logEntry1.getTime()) {
                        return logEntry1;
                    } else return logEntry2;
                }
        ).mapToPair(t -> new Tuple2<String, Long>(t._1(), t._2().getTime()));

        printAfterCalculatingLongestViewingTime(userMaxTimeLogEntryJavaPairRDD);

        // To avoid collecting max time for each user
        JavaPairRDD<String, LogEntry> userIdToRecencyTotalTimeJoinRDDPair = recencyTotalTimeJoinRDDPair.mapToPair(t -> {
            return new Tuple2<String, LogEntry>(t._1().substring(0, t._1().indexOf("_")), t._2());
        });

        JavaPairRDD<String, Tuple2<LogEntry, Long>> join = userIdToRecencyTotalTimeJoinRDDPair.join(userMaxTimeLogEntryJavaPairRDD);

        JavaPairRDD<String, LogEntry> userContentToFinalLogEntriesRDD = join.mapToPair(t -> {
            LogEntry entry = SerializationUtils.clone(t._2()._1());
            entry.setFrequency(t._2()._1().getTime() / t._2()._2());
            return new Tuple2<String, LogEntry>(t._2()._1().getUserId() + "_" + t._2()._1().getContentId(), entry);
        });

        JavaRDD<LogEntry> finalLogEntries = userContentToFinalLogEntriesRDD.map(new Function<Tuple2<String, LogEntry>, LogEntry>() {
            @Override
            public LogEntry call(Tuple2<String, LogEntry> stringLogEntryTuple) throws Exception {
                stringLogEntryTuple._2().setRating(Math.ceil(RATING_FUNCTION.call(stringLogEntryTuple._2().getRecency(), stringLogEntryTuple._2().getFrequency())));
                return stringLogEntryTuple._2();
            }
        });

        printFinalLogEntries(finalLogEntries);

        // Map file to Ratings(user,item,rating) tuples
        JavaRDD<Rating> ratings = finalLogEntries.filter(logEntry -> logEntry.getUserId() != null && logEntry.getContentId() != null).map(logEntry -> {
            return new Rating(Integer.parseInt(logEntry.getUserId()), Integer.parseInt(logEntry.getContentId()), logEntry.getRating());
        });

        printRatingRDD(ratings);

        // Build the recommendation model using ALS
        int rank = 10; // 10 latent factors
        int numIterations = 10; // number of iterations

        MatrixFactorizationModel model = ALS.trainImplicit(JavaRDD.toRDD(ratings),
                rank, numIterations);

        // Create user-item tuples from ratings
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings
                .map(r -> new Tuple2<Object, Object>(r.user(), r.product()));

        JavaPairRDD<Integer, String> itemDescription = fetchItemDescRDD(sparkSession, contentInfoPath);

        printUserProductsRDD(userProducts);
        List<Integer> users = userProducts.map(t -> t._1()).filter(i -> i != null).distinct().map(i -> (int) i).collect();
        users.forEach(u -> recommend(sparkSession, sc, contentInfoPath, model, userProducts, u, itemDescription));
    }

    private static void printUserProductsRDD(JavaRDD<Tuple2<Object, Object>> userProducts) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING USER PRODUCTS RDD");
            userProducts.foreach(i -> System.out.println(String.valueOf(i._1()) + ": " + String.valueOf(i._2())));
        }
    }

    private static void printAfterRecencyCalculation(JavaPairRDD<String, Double> userContentRecencyRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER RECENCY CALCULATION....");
            userContentRecencyRDD.foreach(item -> System.out.println(item._1() + ":" + item._2()));
        }
    }

    private static void printAfterTimeDiffCalculation(JavaPairRDD<String, LogEntry> sessionLogEntryJavaPairRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER TIME DIFF CALCULATION....");
            sessionLogEntryJavaPairRDD.foreach(item -> System.out.println(item._1() + ":" + item._2().getTime()));
        }
    }

    private static void printAfterMappingUserAndContentIdToLogEntry(JavaPairRDD<String, LogEntry> userContentKeyLogEntryPairRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER MAPPING USER , CONTENT TO LOGENTRY ....");
            userContentKeyLogEntryPairRDD.foreach(item -> System.out.println(item._1() + ":" + item._2().getTime()));
        }
    }

    private static void printAfterCalculatingTotalTime(JavaPairRDD<String, LogEntry> totalTimeLogEntryPairRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER CALCULATING TOTAL TIME....");
            totalTimeLogEntryPairRDD.foreach(item -> System.out.println(item._1() + ":" + item._2().getTime()));
        }
    }

    private static void printAfterMappingUserWithViewingTime(JavaPairRDD<String, LogEntry> userLogEntryJavaPairRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER MAPPING USER WITH VIEWING TIME....");
            userLogEntryJavaPairRDD.foreach(item -> System.out.println(item._1() + ":" + item._2().getTime()));
        }
    }

    private static void printAfterCalculatingLongestViewingTime(JavaPairRDD<String, Long> userMaxTimeLogEntryJavaPairRDD) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING AFTER CALCULATING LONGEST VIEWING TIME....");
            userMaxTimeLogEntryJavaPairRDD.foreach(item -> System.out.println(item._1() + ":" + item._2()));
        }
    }

    private static void printFinalLogEntries(JavaRDD<LogEntry> finalLogEntries) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING FINAL LOG ENTRIES....");
            finalLogEntries.foreach(item -> System.out.println(item.getUserId() + "," + item.getContentId() + "," + item.getRating()));
        }
    }

    private static void printRatingRDD(JavaRDD<Rating> ratings) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING RATING RDD....");
            ratings.foreach(item -> System.out.println(item.user() + "," + item.product() + "," + item.rating()));
        }
    }

    public static void recommend(SparkSession sparkSession, JavaSparkContext sc, String contentInfoPath, MatrixFactorizationModel model, JavaRDD<Tuple2<Object, Object>> userProducts, final int userId, final JavaPairRDD<Integer, String> itemDescription) {
        // Calculate the itemIds not rated by a particular user with id give as userId
        JavaRDD<Integer> notRatedByUser = userProducts
                .filter(v1 -> ((Integer) v1._1).intValue() != userId)
                .map(v1 -> (Integer) v1._2);


        // Create user-item tuples for the items that are not rated by user, with given user id
        JavaRDD<Tuple2<Object, Object>> itemsNotRatedByUser = notRatedByUser
                .map(r -> new Tuple2<Object, Object>(userId, r));

        // Predict the ratings of the items not rated by user for the given user
        JavaRDD<Rating> recomondations = model.predict(itemsNotRatedByUser.rdd()).toJavaRDD().distinct();

        // Sort the recommendations by rating in descending order
        recomondations = recomondations.sortBy(v1 -> v1.rating(), false, 1);

        // Get top 10 recommendations
        JavaRDD<Rating> topRecomondations = sc.parallelize(recomondations.take(10));


        // Join top 10 recommendations with item descriptions
        JavaRDD<Tuple2<Rating, String>> recommendedItems = topRecomondations
                .mapToPair(t -> new Tuple2<Integer, Rating>(t.product(), t))
                .join(itemDescription).values();


        //Print the top recommendations for the given user.
        System.out.println("PRINTING RECOMMENDATIONS FOR USER:" + userId);
        recommendedItems.foreach(t -> System.out.println(t._1.product() + "\t" + t._1.rating() + "\t" + t._2));
    }

    public static JavaPairRDD<Integer, String> fetchItemDescRDD(SparkSession sparkSession, String contentInfoPath) {
        // Read item description file. format - itemId, itemName, Other Fields,..
        JavaRDD<String> itemDescritpionFile = sparkSession.read().textFile(contentInfoPath).javaRDD();

        // Create tuples(itemId,ItemDescription), will be used later to get names of item from itemId
        return itemDescritpionFile.mapToPair(t -> {
            String[] s = t.split(",");
            return new Tuple2<Integer, String>(Integer.parseInt(s[0]), s[1]);
        });
    }

    public static void generateLogEntries() {

        final int NUMBER_OF_ENTRIES = 100;
        Calendar calendar = Calendar.getInstance();
        calendar.set(Calendar.DAY_OF_MONTH, Calendar.DAY_OF_MONTH - 1);
        final long LOG_ENTRY_TIME = calendar.getTimeInMillis();
        Random random = new Random();

        StringBuffer logEntries = new StringBuffer();
        try (FileOutputStream fileOutputStream = new FileOutputStream(new File(path))) {
            IntStream.rangeClosed(1, NUMBER_OF_ENTRIES).forEach(i -> {

                int userId = random.nextInt(20);
                StringBuffer entry = new StringBuffer();
                String UUId = UUID.randomUUID().toString();
                int contentId = random.nextInt(100);
                entry.append(UUId).append(",").append(userId).append(",").append(contentId);
                long startTime = LOG_ENTRY_TIME + +random.nextInt(100000);
                entry.append(",").append(startTime);
                entry.append("\n");
                entry.append(UUId).append(",").append(userId).append(",").append(contentId);
                entry.append(",").append(startTime + random.nextInt(100000));
                entry.append("\n");
                logEntries.append(entry);
            });

            fileOutputStream.write(logEntries.toString().getBytes());
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    private static long getTimeDiff(long time1, long time2) {
        Date date1 = new Date(time1);
        Date date2 = new Date(time2);

        if (date2.after(date1))
            return date2.getTime() - date1.getTime();
        else
            return date1.getTime() - date2.getTime();
    }
}
