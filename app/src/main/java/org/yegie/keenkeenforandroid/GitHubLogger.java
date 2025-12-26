/*
 * GitHubLogger.java: Remote crash reporting and analytics
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: Copyright (C) 2024-2025 KeenKeen Contributors
 */

package org.yegie.keenkeenforandroid;

import android.os.Build;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Date;

/**
 * Clever log submission tool using GitHub Gists API.
 */
public class GitHubLogger {
    private static final String TAG = "GitHubLogger";
    private static final String GIST_API = "https://api.github.com/gists";

    public interface LogCallback {
        void onSuccess(String url);
        void onFailure(Exception e);
    }

    public static void collectAndSubmit(String token, LogCallback callback) {
        new Thread(() -> {
            try {
                String logs = captureLogcat();
                String metadata = collectMetadata();
                String content = "# Keen Debug Log\n\n" + metadata + "\n\n## Logs\n```\n" + logs + "\n```";
                
                String gistUrl = postToGist(token, content);
                if (callback != null) callback.onSuccess(gistUrl);
            } catch (Exception e) {
                Log.e(TAG, "Log submission failed", e);
                if (callback != null) callback.onFailure(e);
            }
        }).start();
    }

    private static String captureLogcat() throws Exception {
        Process process = Runtime.getRuntime().exec("logcat -d -v time *:V");
        try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            StringBuilder log = new StringBuilder();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                log.append(line).append("\n");
            }
            return log.toString();
        } finally {
            process.destroy();
        }
    }

    private static String collectMetadata() {
        return "## Device Metadata\n" +
                "* **Model**: " + Build.MODEL + "\n" +
                "* **Manufacturer**: " + Build.MANUFACTURER + "\n" +
                "* **Android Version**: " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n" +
                "* **Architecture**: " + Build.SUPPORTED_ABIS[0] + "\n" +
                "* **Timestamp**: " + new Date().toString() + "\n";
    }

    private static String postToGist(String token, String content) throws Exception {
        URL url = new URL(GIST_API);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Authorization", "token " + token);
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/vnd.github.v3+json");
        conn.setDoOutput(true);

        JSONObject gist = new JSONObject();
        gist.put("description", "Keen Debug Logs - " + Build.MODEL);
        gist.put("public", false); // Secret gist

        JSONObject files = new JSONObject();
        JSONObject fileContent = new JSONObject();
        fileContent.put("content", content);
        files.put("keen_debug_log.md", fileContent);
        
        gist.put("files", files);

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = gist.toString().getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        try {
            int code = conn.getResponseCode();
            if (code >= 200 && code < 300) {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        response.append(line.trim());
                    }
                    JSONObject resJson = new JSONObject(response.toString());
                    return resJson.getString("html_url");
                }
            } else {
                throw new Exception("HTTP Error: " + code);
            }
        } finally {
            conn.disconnect();
        }
    }
}
