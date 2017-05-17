package com.sample.spark.task;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.*;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.codehaus.janino.Java;
import scala.Tuple2;
import scala.Tuple3;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;

/**
 * Created by chandrashekar.v on 5/10/2017.
 */
public class RecommendationEngineWithDataFrame {
    private static final boolean ENABLE_DEBUG_MODE = false;
    private static final int R = 5;
    private static final int C1 = 5;
    private static final int C2 = 5;

    public static void main(String[] args) {

        RecommendationEngine.generateLogEntries();

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("RecommendationEngineWithDataFrame")
                .config("spark.master", "local")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(sparkSession.sparkContext());
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        // playMethod2(sparkSession);

        // Create RDD
        JavaRDD<String> logEntriesRDD = sparkSession.read().textFile(RecommendationEngine.path).javaRDD();

        // Convert RDD to Rows
        JavaRDD<Row> logEntriesRowRDD = logEntriesRDD.map(r -> {
            String[] attributes = r.split(",");
            return RowFactory.create(attributes);
        });

        // Schema
        String schemaString = "sessionId userId contentId timestamp";

        List<StructField> structFields = new ArrayList<>();
        Arrays.stream(schemaString.split(" ")).forEach(field -> structFields.add(DataTypes.createStructField(field, DataTypes.StringType, true)));

        StructType schema = DataTypes.createStructType(structFields);

        //Apply the schema to the row RDD.
        Dataset<Row> logEntriesDFFromRDD = sparkSession.createDataFrame(logEntriesRowRDD, schema);

        // playMethod1(logEntriesDFFromRDD);

        logEntriesDFFromRDD.createOrReplaceTempView("log");

        // playMethod(sparkSession);


        // Viewing time for each content based on session ID
        Dataset<Row> viewingTimeDF = sparkSession.sql("SELECT log1.sessionId, log1.userId, log1.contentId" +
                ", (log2.timestamp-log1.timestamp) as timediff FROM log log1 INNER JOIN log log2 on log1.sessionId=log2.sessionId where (log2.timestamp-log1.timestamp) > 0").toDF();

        printViewingTimeDF(viewingTimeDF);

        Dataset<Row> userContentTotalTimeDF = viewingTimeDF
                .groupBy("userId", "contentId")
                .sum("timediff").orderBy("userId").toDF("userId", "contentId", "timediff");

        printUserContentAndTimeDiffDF(userContentTotalTimeDF);

        // Calculate Recency.
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

        recencyDF.createOrReplaceTempView("recencyDFView");
        List<Row> longestViewingTimebyUserDF = viewingTimeDF.groupBy("userId").max("timediff").toDF("userId", "longest_view_time").collectAsList();

        Map<String, Double> map = new HashMap<>();
        longestViewingTimebyUserDF.stream().forEach(row -> map.put(row.getAs("userId"), (Double) row.getAs("longest_view_time")));
        Dataset<Row> frequencyDF = userContentTotalTimeDF.map((MapFunction<Row, Tuple3<String, String, Double>>) row -> {
            double t1 = (double) row.getAs("timediff");
            String userId = (String) row.getAs("userId");
            double timediff1 = map.get(userId);
            return new Tuple3<String, String, Double>(row.getAs("userId"), row.getAs("contentId"), C2 * (t1 / timediff1));

        }, Encoders.tuple(Encoders.STRING(), Encoders.STRING(), Encoders.DOUBLE())).toDF("userId", "contentId", "frequency");

        printFrequencyDF(frequencyDF);

        frequencyDF.createOrReplaceTempView("frequencyDFView");

        Dataset<Row> userContentRatingDF = sparkSession.sql("SELECT r.userId, r.contentId, r.recency+f.frequency " +
                "from recencyDFView r, frequencyDFView f where r.userId = f.userId AND r.contentId = f.contentId").toDF("userId", "contentId", "rating");

        printUserContentRatingDF(userContentRatingDF);

        JavaRDD<Row> userContentRatingRDD = userContentRatingDF.toJavaRDD();
        JavaRDD<Rating> ratingsJavaRDD = userContentRatingRDD.map(row -> {
            return new Rating(Integer.parseInt(row.getAs("userId")), Integer.parseInt(row.getAs("contentId")), row.getAs("rating"));
        });

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


    private static void playMethod2(SparkSession sparkSession) {
        Dataset<String> logEntriesDF = sparkSession.read().textFile(RecommendationEngine.path);

        logEntriesDF.show(10, false);

        logEntriesDF.printSchema();
    }

    private static void playMethod1(Dataset<Row> logEntriesDFFromRDD) {
        logEntriesDFFromRDD.printSchema();
        logEntriesDFFromRDD.show(10, false);
        logEntriesDFFromRDD
                .select(col("sessionId"), col("userId"), col("contentId"), col("timestamp"))
                .show(10, false);
    }

    private static void playMethod(SparkSession sparkSession) {
        // Running SQL over a temperory view created using DataFrames.
        Dataset<Row> allUsers = sparkSession.sql("SELECT userId from log");
        allUsers.show(false);

        /* Make use of distinct() on returned results from above sql query. */
        allUsers.distinct().show(false);

        /* The results of the SQL queries are DataFrames and support all the normal RDD operations.
         The columns of the row in the result can be accessed by field index or field name.
         */
        Dataset<String> userIdDS = allUsers.map((MapFunction<Row, String>) r -> "User: " + r.getString(0), Encoders.STRING());

        // Showing user ids using userIdDS create using SQL query result allUsers
        userIdDS.distinct().show(false);
    }
}
