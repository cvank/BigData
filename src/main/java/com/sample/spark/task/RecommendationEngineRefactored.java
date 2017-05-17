package com.sample.spark.task;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.Date;

/**
 * Created by chandrashekar.v on 5/8/2017.
 */
public class RecommendationEngineRefactored {


    private static final int R = 10;
    private static final int C1 = 10;
    private static final int C2 = 10;

    static Function2<Double, Double, Double> ratingFunction = (recency, frequency) -> (C1 * recency) + (C2 * frequency);

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("WordCount").config("spark.master", "local").getOrCreate();
        String path = "D:\\Work\\Bigdata\\Spark-2.1.0\\spark-2.1.0-bin-hadoop2.7\\data\\mllib\\als\\test.data";
        JavaRDD<String> data = sparkSession.read().textFile(path).javaRDD();

        JavaPairRDD<String, LogEntry> totalTimeLogEntryPairRDD = data.mapToPair(e -> {
            return createLogEntries(e);
        }).reduceByKey((entry1, entry2) -> {
            return calculateTimeDiffOnLogEntries(entry1, entry1.getTime(), entry2.getTime());
        }).mapToPair(logEntryTuple -> {
            return mapByUserAndContentId(logEntryTuple);
        }).reduceByKey((entry1, entry2) -> {
            return addTotalViewingTime(entry1, entry2);
        });

        JavaPairRDD<String, Long> userContentRecencyRDD = data.mapToPair(e -> {
            String[] tokens = e.split(",");
            long timeElapsed = System.currentTimeMillis() - Long.valueOf(tokens[3]);
            return new Tuple2<String, Long>(tokens[1] + "_" + tokens[2], Long.valueOf(tokens[3]));
        }).reduceByKey((aLong, aLong2) -> Math.min(aLong, aLong2)).mapToPair(t -> new Tuple2<String, Long>(t._1, -R * t._2()));

        // To find maximum difference
        JavaPairRDD<String, LogEntry> userMaxTimeLogEntryJavaPairRDD = totalTimeLogEntryPairRDD.mapToPair(new PairFunction<Tuple2<String, LogEntry>, String, LogEntry>() {
            @Override
            public Tuple2<String, LogEntry> call(Tuple2<String, LogEntry> logEntryTuple) throws Exception {
                return new Tuple2<String, LogEntry>(logEntryTuple._2().getUserId(), logEntryTuple._2());
            }
        }).reduceByKey((logEntry1, logEntry2) -> {
                    if (Long.compare(logEntry1.getTime(), logEntry2.getTime()) > 0)
                        return logEntry1;
                    else return logEntry2;
                }
        );

        JavaRDD<LogEntry> map = totalTimeLogEntryPairRDD.map(new Function<Tuple2<String, LogEntry>, LogEntry>() {
            @Override
            public LogEntry call(Tuple2<String, LogEntry> stringLogEntryTuple) throws Exception {
                Tuple2<String, LogEntry> first = userMaxTimeLogEntryJavaPairRDD.filter(t -> t._1().equals(stringLogEntryTuple._2().getUserId())).first();
                stringLogEntryTuple._2().setFrequency(stringLogEntryTuple._2().getTime() / first._2().getTime());
                Tuple2<String, Long> recencyTuple = userContentRecencyRDD.filter(e -> e._1().equals(stringLogEntryTuple._1())).first();
                stringLogEntryTuple._2().setRecency(recencyTuple._2());
                stringLogEntryTuple._2().setRating(ratingFunction.call(stringLogEntryTuple._2().getRecency(), stringLogEntryTuple._2().getFrequency()));
                return stringLogEntryTuple._2();
            }
        });
    }

    private static LogEntry addTotalViewingTime(LogEntry entry1, LogEntry entry2) {
        long totalTime = entry1.getTime() + entry2.getTime();
        LogEntry logEntry1 = new LogEntry();
        logEntry1.setTime(totalTime);
        return logEntry1;
    }

    private static Tuple2<String, LogEntry> mapByUserAndContentId(Tuple2<String, LogEntry> logEntryTuple) {
        final String key = logEntryTuple._2().getUserId() + "_" + logEntryTuple._2().getContentId();
        return new Tuple2<String, LogEntry>(key, logEntryTuple._2());
    }

    private static LogEntry calculateTimeDiffOnLogEntries(LogEntry entry1, long time, long time2) {
        LogEntry result = new LogEntry();
        result = entry1;
        long diff = getTimeDiff(time, time2);
        result.setTime(diff);
        return result;
    }

    private static Tuple2<String, LogEntry> createLogEntries(String e) {
        String[] tokens = e.split(",");
        LogEntry entry = new LogEntry();
        entry.setSessionId(tokens[0]);
        entry.setUserId(tokens[1]);
        entry.setContentId(tokens[2]);
        entry.setTime(Long.valueOf(tokens[3]));
        entry.setContentType(tokens[4]);
        return new Tuple2<String, LogEntry>(tokens[0], entry);
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
