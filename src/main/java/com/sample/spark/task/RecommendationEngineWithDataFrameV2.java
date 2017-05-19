package com.sample.spark.task;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;
import scala.Tuple3;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.apache.spark.sql.functions.col;

/**
 * Created by chandrashekar.v on 5/10/2017.
 */
public class RecommendationEngineWithDataFrameV2 {
    private static final boolean ENABLE_DEBUG_MODE = false;
    private static final int R = 5;
    private static final int C1 = 5;
    private static final int C2 = 5;

    public static void main(String[] args) {

        RecommendationEngine.generateLogEntries();

        SparkSession sparkSession = getSparkSession();

        JavaSparkContext sc = getJavaSparkContext(sparkSession);

        setLoggerLevel();

        // Create RDD
        JavaRDD<String> logEntriesRDD = readLogFile(sparkSession);

        // Convert RDD to Rows
        JavaRDD<Row> logEntriesRowRDD = convertRDDToRows(logEntriesRDD);

        // Schema
        String schemaString = getSchemaString();

        StructType schema = createSchema(schemaString);

        //Apply the schema to the row RDD.
        Dataset<Row> logEntriesDFFromRDD = applySchemaToRDD(sparkSession, logEntriesRowRDD, schema);

        logEntriesDFFromRDD.createOrReplaceTempView("log");

        // Viewing time for each content based on session ID
        Dataset<Row> viewingTimeDF = getViewingTimeDF(sparkSession);

        // Calculate Total viewing time for user and content combination.
        Dataset<Row> userContentTotalTimeDF = getTotalTimeByUserandContent(viewingTimeDF);

        // Calculate Recency.
        Dataset<Row> recencyDF = calculateRecency(sparkSession);

        // Calculate Longest viewing time for each user
        Dataset<Row> longestViewingTimeDF = getLongestViewingTimeForEachUserAsDF(viewingTimeDF);

        // Create UDF to calculate Frequency
        registerFrequencyUDF(sparkSession);

        // Join on User content total time and Longest viewing time per user.
        Dataset<Row> userContentTotalTimeAndLongestTimeJoinDF = joinUserContentTotalTimeAndLongestViewingTime(userContentTotalTimeDF, longestViewingTimeDF);

        // Calculate Frequency by using FREQUENCY UDF.
        Dataset<Row> frequencyDF = calculateFrequency(userContentTotalTimeAndLongestTimeJoinDF);


        // Register Rating UDF
        registerRatingUDF(sparkSession);

        // create ratings for each user and content making use of recency and frequency.
        Dataset<Row> ratingDF = getRatingDF(recencyDF, frequencyDF);

        // Convert Rating DF to Java RDD
        JavaRDD<Rating> ratingsJavaRDD = convertRatingDFtoJavaRDD(ratingDF);

        // Perform ALS and print recommendations.
        triggerALS(sparkSession, sc, ratingsJavaRDD);
    }

    private static void registerRatingUDF(SparkSession sparkSession) {
        sparkSession.udf().register("RATING", new UDF2<Double, Double, Double>() {
            @Override
            public Double call(Double aDouble, Double aDouble2) throws Exception {
                return aDouble + aDouble2;
            }
        }, DataTypes.DoubleType);
    }

    private static void triggerALS(SparkSession sparkSession, JavaSparkContext sc, JavaRDD<Rating> ratingsJavaRDD) {
        int rank = 10; // 10 latent factors
        int numIterations = 10; // number of iterations

        System.out.println("Trigger ALS");
        MatrixFactorizationModel model = ALS.trainImplicit(ratingsJavaRDD.rdd(), rank, numIterations);

        JavaRDD<Tuple2<Object, Object>> userProducts = ratingsJavaRDD.map(r -> new Tuple2<Object, Object>(r.user(), r.product()));
        List<Integer> users = userProducts.map(t -> t._1()).filter(i -> i != null).distinct().map(i -> (int) i).collect();

        System.out.println("Calling Recommend method");
        JavaPairRDD<Integer, String> itemDescription = RecommendationEngine.fetchItemDescRDD(sparkSession, RecommendationEngine.contentInfoPath);
        users.forEach(u -> RecommendationEngine.recommend(sparkSession, sc, RecommendationEngine.contentInfoPath, model, userProducts, u, itemDescription));
    }

    private static JavaRDD<Rating> convertRatingDFtoJavaRDD(Dataset<Row> ratingDF) {
        JavaRDD<Row> userContentRatingRDD = ratingDF.toJavaRDD();
        return userContentRatingRDD.map(row -> {
            return new Rating(Integer.parseInt(row.getAs("userId")), Integer.parseInt(row.getAs("contentId")), row.getAs("rating"));
        });
    }

    private static Dataset<Row> getRatingDF(Dataset<Row> recencyDF, Dataset<Row> frequencyDF) {
        return frequencyDF.join(recencyDF, frequencyDF.col("userId").equalTo(recencyDF.col("userId")).and(frequencyDF.col("contentId").equalTo(recencyDF.col("contentId"))))
                .select(frequencyDF.col("userId"), frequencyDF.col("contentId"), col("frequency"), col("recency"))
                .withColumn("rating", callUDF("RATING", col("frequency"), col("recency")))
                .select("userId", "contentId", "rating").toDF("userId", "contentId", "rating");
    }

    private static Dataset<Row> calculateFrequency(Dataset<Row> userContentTotalTimeAndLongestTimeJoinDF) {
        return userContentTotalTimeAndLongestTimeJoinDF.select("userId", "contentId", "timediff", "longest_view_time")
                .withColumn("frequency", callUDF("FREQUENCY", col("timediff"), col("longest_view_time")));
    }

    private static Dataset<Row> joinUserContentTotalTimeAndLongestViewingTime(Dataset<Row> userContentTotalTimeDF, Dataset<Row> longestViewingTimeDF) {
        return userContentTotalTimeDF.join(longestViewingTimeDF, "userId");
    }

    private static void registerFrequencyUDF(SparkSession sparkSession) {
        sparkSession.udf().register("FREQUENCY", new UDF2<Double, Double, Double>() {
            @Override
            public Double call(Double aDouble, Double aDouble2) throws Exception {
                return C2 * (aDouble / aDouble2);
            }
        }, DataTypes.DoubleType);
    }

    private static Dataset<Row> getLongestViewingTimeForEachUserAsDF(Dataset<Row> viewingTimeDF) {
        return viewingTimeDF.groupBy("userId").max("timediff").toDF("userId", "longest_view_time");
    }

    private static Dataset<Row> getTotalTimeByUserandContent(Dataset<Row> viewingTimeDF) {
        Dataset<Row> userContentTotalTimeDF = viewingTimeDF
                .groupBy("userId", "contentId")
                .sum("timediff").orderBy("userId").toDF("userId", "contentId", "timediff");

        printUserContentAndTimeDiffDF(userContentTotalTimeDF);
        return userContentTotalTimeDF;
    }

    private static Dataset<Row> getViewingTimeDF(SparkSession sparkSession) {
        Dataset<Row> viewingTimeDF = sparkSession.sql("SELECT log1.sessionId, log1.userId, log1.contentId" +
                ", (log2.timestamp-log1.timestamp) as timediff FROM log log1 INNER JOIN log log2 on log1.sessionId=log2.sessionId where (log2.timestamp-log1.timestamp) > 0").toDF();

        printViewingTimeDF(viewingTimeDF);
        return viewingTimeDF;
    }

    private static Dataset<Row> applySchemaToRDD(SparkSession sparkSession, JavaRDD<Row> logEntriesRowRDD, StructType schema) {
        return sparkSession.createDataFrame(logEntriesRowRDD, schema);
    }

    private static StructType createSchema(String schemaString) {
        List<StructField> structFields = new ArrayList<>();
        Arrays.stream(schemaString.split(" ")).forEach(field -> structFields.add(DataTypes.createStructField(field, DataTypes.StringType, true)));

        return DataTypes.createStructType(structFields);
    }

    private static String getSchemaString() {
        return "sessionId userId contentId timestamp";
    }

    private static JavaRDD<Row> convertRDDToRows(JavaRDD<String> logEntriesRDD) {
        return logEntriesRDD.map(r -> {
            String[] attributes = r.split(",");
            return RowFactory.create(attributes);
        });
    }

    private static JavaRDD<String> readLogFile(SparkSession sparkSession) {
        return sparkSession.read().textFile(RecommendationEngine.path).javaRDD();
    }

    private static void setLoggerLevel() {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
    }

    private static JavaSparkContext getJavaSparkContext(SparkSession sparkSession) {
        return new JavaSparkContext(sparkSession.sparkContext());
    }

    private static SparkSession getSparkSession() {
        return SparkSession
                .builder()
                .appName("RecommendationEngineWithDataFrame")
                .config("spark.master", "local")
                .getOrCreate();
    }

    private static Dataset<Row> calculateRecency(SparkSession sparkSession) {
        Dataset<Row> allLogEntriesDF = sparkSession.sql("SELECT userId, contentId,timestamp from log");
        Dataset<Row> userContentMaxTimestampDF = allLogEntriesDF
                .withColumn("timestamp", col("timestamp").cast("long"))
                .groupBy("userId", "contentId").max("timestamp").toDF("userId", "contentId", "max_times_tamp");

        Dataset<Row> recencyDF = userContentMaxTimestampDF.orderBy("userId")
                .map((MapFunction<Row, Tuple3<String, String, Double>>) row -> {
                    long maxTime = row.getAs("max_times_tamp");
                    double recency = Math.exp(-R * TimeUnit.MILLISECONDS.toDays(System.currentTimeMillis() - maxTime));
                    return new Tuple3<String, String, Double>(row.getAs("userId"), row.getAs("contentId"), C1 * recency);

                }, Encoders.tuple(Encoders.STRING(), Encoders.STRING(), Encoders.DOUBLE())).toDF("userId", "contentId", "recency");

        printRecencyDF(recencyDF);
        return recencyDF;
    }

    private static void printUserContentRatingDF(Dataset<Row> userContentRatingDF) {
        if (ENABLE_DEBUG_MODE)
            userContentRatingDF.show(false);
    }

    private static void printFrequencyDF(Dataset<Row> frequencyDF) {
        if (ENABLE_DEBUG_MODE)
            frequencyDF.show();
    }

    private static void printRecencyDF(Dataset<Row> recencyDF) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING RECENCY DF.");
            recencyDF.show();
        }
    }

    private static void printUserContentAndTimeDiffDF(Dataset<Row> userContentTotalTimeDF) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING USER CONTENT AND TIME DIFF DATA FRAME");
            userContentTotalTimeDF.show(false);
        }
    }

    private static void printViewingTimeDF(Dataset<Row> viewingTimeDF) {
        if (ENABLE_DEBUG_MODE) {
            System.out.println("PRINTING Viewing Time DF");
            viewingTimeDF.show(false);
        }
    }
}
