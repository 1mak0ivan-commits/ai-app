// netlify/functions/chat.js
// Serverless backend — runs on Netlify as AWS Lambda (Node 18+)

"use strict";

const MODELS = {
  gpt:      "openai/gpt-4o-mini",
  claude:   "anthropic/claude-3-haiku",
  deepseek: "deepseek/deepseek-chat",
  llama:    "meta-llama/llama-3-70b-instruct",
};

const DEFAULT_SYSTEM = (
  "Ты умный и дружелюбный AI-ассистент. " +
  "Отвечай полезно, чётко и по делу. " +
  "Используй markdown для форматирования: **жирный**, `код`, ```блоки кода```."
);

const cors = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

function json(status, body) {
  return {
    statusCode: status,
    headers: { "Content-Type": "application/json", ...cors },
    body: JSON.stringify(body),
  };
}

exports.handler = async (event) => {
  // CORS preflight
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers: cors, body: "" };
  }

  // Health check
  if (event.httpMethod === "GET") {
    return json(200, { status: "ok", models: Object.keys(MODELS) });
  }

  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed" });
  }

  // Parse body
  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Invalid JSON body" });
  }

  const { text, model = "gpt", memory = [], persona = "" } = body;

  if (!text || !String(text).trim()) {
    return json(400, { error: "Field 'text' is required" });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error("OPENROUTER_API_KEY env var not set");
    return json(500, { error: "Server not configured" });
  }

  const modelId  = MODELS[String(model).toLowerCase()] || MODELS.gpt;
  const sysPrompt = persona ? String(persona).trim() : DEFAULT_SYSTEM;

  // Build messages — system + last 16 history + new user message
  const messages = [
    { role: "system", content: sysPrompt },
    ...memory.slice(-16),
    { role: "user", content: String(text).trim() },
  ];

  // Call OpenRouter
  let orResponse;
  try {
    orResponse = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${apiKey}`,
        "HTTP-Referer":  "https://curious-pudd.netlify.app",
        "X-Title":       "AI Assistant",
      },
      body: JSON.stringify({
        model:       modelId,
        messages,
        temperature: 0.7,
        max_tokens:  2048,
      }),
    });
  } catch (err) {
    console.error("Network error calling OpenRouter:", err);
    return json(503, { error: "Could not reach AI service" });
  }

  if (!orResponse.ok) {
    const errBody = await orResponse.text().catch(() => "");
    console.error(`OpenRouter ${orResponse.status}:`, errBody);
    return json(502, {
      error: `AI service error (${orResponse.status})`,
      detail: errBody.slice(0, 300),
    });
  }

  let orData;
  try {
    orData = await orResponse.json();
  } catch {
    return json(502, { error: "Invalid response from AI service" });
  }

  const answer = orData.choices?.[0]?.message?.content?.trim() || "";
  if (!answer) {
    return json(502, { error: "Empty response from model" });
  }

  // Return answer + updated memory
  return json(200, {
    answer,
    model: modelId,
    memory: [
      ...memory,
      { role: "user",      content: String(text).trim() },
      { role: "assistant", content: answer },
    ],
  });
};
