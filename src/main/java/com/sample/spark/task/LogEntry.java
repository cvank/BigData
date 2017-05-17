package com.sample.spark.task;

import java.io.Serializable;
import java.util.Date;

/**
 * Created by chandrashekar.v on 5/8/2017.
 */
public class LogEntry implements Serializable {
    private String userId;
    private String contentId;
    private long time;
    private String contentType;
    private String sessionId;
    private double frequency;
    private long noOfDaysElapsed;
    private double rating;
    private double recency;

    public double getRating() {
        return rating;
    }

    public void setRating(double rating) {
        this.rating = rating;
    }

    public double getRecency() {
        return recency;
    }

    public void setRecency(double recency) {
        this.recency = recency;
    }

    public long getNoOfDaysElapsed() {
        return noOfDaysElapsed;
    }

    public void setNoOfDaysElapsed(long noOfDaysElapsed) {
        this.noOfDaysElapsed = noOfDaysElapsed;
    }

    public double getFrequency() {
        return frequency;
    }

    public void setFrequency(double frequency) {
        this.frequency = frequency;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getContentId() {
        return contentId;
    }

    public void setContentId(String contentId) {
        this.contentId = contentId;
    }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }
}
