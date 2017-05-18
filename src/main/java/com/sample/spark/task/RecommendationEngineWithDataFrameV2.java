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

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("RecommendationEngineWithDataFrame")
                .config("spark.master", "local")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(sparkSession.sparkContext());
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

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

        logEntriesDFFromRDD.createOrReplaceTempView("log");

        // Viewing time for each content based on session ID
        Dataset<Row> viewingTimeDF = sparkSession.sql("SELECT log1.sessionId, log1.userId, log1.contentId" +
                ", (log2.timestamp-log1.timestamp) as timediff FROM log log1 INNER JOIN log log2 on log1.sessionId=log2.sessionId where (log2.timestamp-log1.timestamp) > 0").toDF();

        printViewingTimeDF(viewingTimeDF);

        Dataset<Row> userContentTotalTimeDF = viewingTimeDF
                .groupBy("userId", "contentId")
                .sum("timediff").orderBy("userId").toDF("userId", "contentId", "timediff");

        printUserContentAndTimeDiffDF(userContentTotalTimeDF);

        // Calculate Recency.
        Dataset<Row> recencyDF = calculateRecency(sparkSession);
        recencyDF.createOrReplaceTempView("recencyDFView");

        Dataset<Row> longestViewingTimeDF = viewingTimeDF.groupBy("userId").max("timediff").toDF("userId", "longest_view_time");

        sparkSession.udf().register("FREQUENCY", new UDF2<Double, Double, Double>() {
            @Override
            public Double call(Double aDouble, Double aDouble2) throws Exception {
                return aDouble / aDouble2;
            }
        }, DataTypes.DoubleType);

        Dataset<Row> userContentTotalTimeAndLongestTimeJoinDF = userContentTotalTimeDF.join(longestViewingTimeDF, "userId");
        Dataset<Row> frequencyDF1 = userContentTotalTimeAndLongestTimeJoinDF.select("userId", "contentId", "timediff", "longest_view_time")
                .withColumn("frequency", callUDF("FREQUENCY", col("timediff"), col("longest_view_time")));

        Dataset<Row> ratingDF = frequencyDF1.join(recencyDF).select("userId", "contentId", "frequency", "recency")
                .withColumn("rating", callUDF("RATING", col("frequency"), col("recency")));

        JavaRDD<Row> userContentRatingRDD = ratingDF.toJavaRDD();
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
