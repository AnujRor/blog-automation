import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const KEYWORDS_PATH = path.join(ROOT, "config", "keywords.json");
const PUBLISHED_PATH = path.join(ROOT, "published.json");

const REQUIRED_ENV = [
  "GOOGLE_CLIENT_ID",
  "GOOGLE_CLIENT_SECRET",
  "GOOGLE_REFRESH_TOKEN",
  "BLOGGER_BLOG_ID",
  "PIXABAY_API_KEY",
  "SERPAPI_KEY",
  "GROQ_API_KEY",
  "AUTHOR_NAME",
  "AUTHOR_BIO",
  "PORTFOLIO_URL",
  "AUTHOR_SOCIALS",
];

function fail(message) {
  console.error("ERROR: " + message);
  process.exit(1);
}

for (const key of REQUIRED_ENV) {
  if (!process.env[key]) fail(`Missing env variable ${key} in .env file.`);
}

const {
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  GOOGLE_REFRESH_TOKEN,
  BLOGGER_BLOG_ID,
  PIXABAY_API_KEY,
  SERPAPI_KEY,
  GROQ_API_KEY,
  AUTHOR_NAME,
  AUTHOR_BIO,
  PORTFOLIO_URL,
  AUTHOR_SOCIALS,
  PUBLISH_MODE,
} = process.env;

const PUBLISH_LIVE = (PUBLISH_MODE || "draft").toLowerCase() === "live";

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + "\n", "utf8");
}

function shuffle(list) {
  const arr = [...list];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function extractJson(text) {
  let cleaned = String(text).trim();
  // Strip markdown code fences wherever they appear.
  cleaned = cleaned.replace(/```(?:json)?/gi, "").trim();

  // Slice out the first complete JSON object (balanced braces, string-aware).
  const sliceBalanced = (str, openChar, closeChar) => {
    const start = str.indexOf(openChar);
    if (start === -1) return null;
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < str.length; i++) {
      const ch = str[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === openChar) depth++;
      else if (ch === closeChar) {
        depth--;
        if (depth === 0) return str.slice(start, i + 1);
      }
    }
    return null;
  };

  const objectSlice = sliceBalanced(cleaned, "{", "}");
  if (objectSlice) {
    try {
      return JSON.parse(objectSlice);
    } catch {
      /* fall through to array attempt */
    }
  }
  const arraySlice = sliceBalanced(cleaned, "[", "]");
  if (arraySlice) {
    return JSON.parse(arraySlice);
  }
  throw new Error("No valid JSON object found in model response.");
}

async function fetchJson(url, options = {}, label = "API") {
  let res;
  try {
    res = await fetch(url, options);
  } catch (err) {
    throw new Error(`${label} network error: ${err.message}`);
  }
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail =
      (data && (data.error_description || data.error?.message || data.error)) ||
      text.slice(0, 300);
    const error = new Error(`${label} failed (HTTP ${res.status}): ${detail}`);
    error.status = res.status;
    error.body = data;
    throw error;
  }
  return data;
}

// ===================== GROQ =====================

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Groq free tier limits tokens per minute (TPM). We keep a rolling 60s window
// of (input + reserved output) tokens so we never trip the limit.
const TOKEN_WINDOW_MS = 60000;
const SAFE_TPM = 5500;
let tokenWindow = [];

async function throttleTokens(tokens) {
  tokenWindow = tokenWindow.filter((t) => Date.now() - t.time < TOKEN_WINDOW_MS);
  const used = tokenWindow.reduce((a, t) => a + t.tokens, 0);
  if (used + tokens > SAFE_TPM && tokenWindow.length > 0) {
    const oldest = tokenWindow[0];
    const waitMs = TOKEN_WINDOW_MS - (Date.now() - oldest.time) + 1000;
    console.log(
      "Rate limit: waiting " + Math.ceil(waitMs / 1000) + "s to respect Groq tokens/min limit..."
    );
    await sleep(waitMs);
    return throttleTokens(tokens);
  }
  tokenWindow.push({ time: Date.now(), tokens });
}

let chatModels = null;
let modelCursor = 0;
const jsonModeOff = new Set();

function isReasoningModel(id) {
  return /gpt-oss|deepseek|reasoning|r1/i.test(id);
}

async function loadGroqChatModels() {
  if (chatModels) return chatModels;
  const data = await fetchJson(
    "https://api.groq.com/openai/v1/models",
    { headers: { Authorization: `Bearer ${GROQ_API_KEY}` } },
    "Groq model list"
  );
  const ids = (Array.isArray(data.data) ? data.data : [])
    .map((m) => m.id)
    .filter((id) => typeof id === "string" && id.length > 0)
    .filter((id) => !/whisper|guard|tts|stt|audio|orpheus|embedding|rerank/i.test(id));
  if (ids.length === 0) {
    throw new Error("No usable chat model available on your Groq account.");
  }
  const rank = (id) => {
    if (id.includes("llama-3.3-70b")) return 0;
    if (id.includes("versatile")) return 1;
    if (id.toLowerCase().includes("llama")) return 2;
    if (id.includes("gpt-oss-120b")) return 3;
    if (id.includes("qwen")) return 4;
    if (id.includes("gpt-oss-20b")) return 5;
    return 6;
  };
  chatModels = ids.slice().sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  console.log("Groq chat models available (in order): " + chatModels.join(", "));
  return chatModels;
}

function currentModel() {
  return chatModels[modelCursor % chatModels.length];
}

async function callGroq(model, messages, maxTokens) {
  const inputChars = messages.reduce((a, m) => a + String(m.content || "").length, 0);
  const inputTokens = Math.ceil(inputChars / 4) + 60;
  const reasoning = isReasoningModel(model);
  // Reasoning models need extra room so reasoning tokens never eat the answer.
  const outputTokens = reasoning ? Math.max(maxTokens, 2048) : maxTokens;
  await throttleTokens(inputTokens + outputTokens);

  const body = {
    model,
    messages,
    temperature: 0.4,
    max_tokens: outputTokens,
  };
  if (!jsonModeOff.has(model)) body.response_format = { type: "json_object" };
  if (reasoning) body.reasoning_effort = "low";

  let res;
  try {
    res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error("Groq network error: " + err.message);
  }
  const text = await res.text();
  return { status: res.status, ok: res.ok, text, headers: res.headers };
}

function serverWaitMs(text, headers) {
  const match = text.match(/try again in ([\d.]+)s/i) || text.match(/in ([\d.]+)s/i);
  const headerRetry = parseFloat((headers && headers.get && headers.get("retry-after")) || "0");
  return Math.max(
    match ? parseFloat(match[1]) * 1000 : 0,
    headerRetry ? headerRetry * 1000 : 0,
    5000
  );
}

const JSON_FIX_INSTRUCTION =
  "Your previous reply was not valid JSON. Return only raw JSON, no markdown, no code fences, no extra text.";

// Calls Groq and always returns a parsed JSON object. Retries invalid JSON and
// rate limits, and falls back to the next available model if one keeps failing.
async function groqJson(messages, maxTokens = 2000) {
  await loadGroqChatModels();
  let lastError = null;

  for (let modelTry = 0; modelTry < chatModels.length; modelTry++) {
    const model = currentModel();
    let msgs = messages.slice();

    for (let attempt = 1; attempt <= 4; attempt++) {
      const { status, ok, text, headers } = await callGroq(model, msgs, maxTokens);

      if (status === 429) {
        const waitMs = serverWaitMs(text, headers);
        console.log(
          `Groq rate limit on ${model} (attempt ${attempt}) — waiting ${Math.ceil(waitMs / 1000)}s...`
        );
        await sleep(waitMs + 1000);
        continue;
      }

      if (status === 400 && /response_format|json_object|json mode/i.test(text)) {
        jsonModeOff.add(model);
        console.log(`Groq model ${model} rejected JSON mode — retrying without it...`);
        continue;
      }

      if (status === 400 && /json/i.test(text)) {
        jsonModeOff.add(model);
        msgs = [...messages, { role: "user", content: JSON_FIX_INSTRUCTION }];
        console.log(
          `Groq returned invalid JSON on ${model} (attempt ${attempt}) — retrying with stricter instruction...`
        );
        await sleep(1500);
        continue;
      }

      if (!ok) {
        lastError = new Error(`Groq (${model}) failed HTTP ${status}: ${text.slice(0, 200)}`);
        if (status === 404 || /model|decommission|not found|not support/i.test(text)) {
          console.log(`Groq model ${model} unavailable — trying next model...`);
          break;
        }
        throw lastError;
      }

      let content = null;
      try {
        const data = JSON.parse(text);
        content = data?.choices?.[0]?.message?.content;
      } catch {
        content = null;
      }
      if (!content) {
        lastError = new Error(`Groq (${model}) returned an empty response.`);
        await sleep(1000);
        continue;
      }

      try {
        return extractJson(content);
      } catch (err) {
        jsonModeOff.add(model);
        lastError = err;
        msgs = [...messages, { role: "user", content: JSON_FIX_INSTRUCTION }];
        console.log(`Groq JSON parse failed on ${model} (attempt ${attempt}) — retrying...`);
        await sleep(1500);
      }
    }

    modelCursor++;
  }

  throw new Error(
    "Groq failed on all available models: " + (lastError ? lastError.message : "unknown error")
  );
}

// ===================== STEP 1: TOPIC DISCOVERY =====================

async function fetchNewsHeadlines(keyword) {
  const url =
    "https://serpapi.com/search.json?engine=google_news" +
    "&hl=en&gl=us&num=10" +
    "&q=" + encodeURIComponent(keyword) +
    "&api_key=" + encodeURIComponent(SERPAPI_KEY);
  const data = await fetchJson(url, {}, "SerpAPI google_news");
  const news = Array.isArray(data.news_results) ? data.news_results : [];
  return news
    .filter((item) => item && typeof item.title === "string" && item.title.trim())
    .slice(0, 10)
    .map((item) => ({
      title: item.title.trim(),
      link: item.link || "",
      source: item.source || "",
      date: item.date || "",
      snippet: item.snippet || "",
    }));
}

async function chooseTopics(headlinesByKeyword, publishedTopics) {
  const flat = [];
  for (const [keyword, items] of Object.entries(headlinesByKeyword)) {
    for (const it of items) flat.push(`[${keyword}] ${it.title} (${it.source || "news"})`);
  }
  if (flat.length === 0) return [];

  const prompt = `You are the topic selector for a Hinglish blog covering ONLY:
1) AI: new AI tools, AI models, chatbots, AI news
2) Tech: AI-assisted coding, web development, programming, automation/APIs, new software or tech products
3) Digital marketing with AI: AI tools for SEO, social media, content marketing, ads, email, growth
FORBIDDEN: gaming, politics, celebrity, sports, crime, health, finance tips, entertainment, and marketing with no AI/tech tool angle.

Headlines:
${flat.join("\n")}

Already published (avoid these and near-duplicates):
${publishedTopics.length ? publishedTopics.join("\n") : "(none yet)"}

Pick up to 3 of the BEST concrete topics, best first. Each must fit an area and be clearly covered by at least one headline naming a real tool, model, product, feature or company action (e.g. "Krea 3 AI image editor", "Tencent open-source coding model"). IGNORE headlines that don't fit and use the ones that do.
Return [] only if NO headline names any AI/tech tool, model, product or feature.

Reply ONLY JSON: {"topics":[{"topic":"...","area":"...","reason":"..."}]} (or {"topics":[]}).`;

  const result = await groqJson(
    [
      {
        role: "system",
        content: "You are a topic selector for an AI/tech Hinglish blog. Reply only with valid JSON.",
      },
      { role: "user", content: prompt },
    ],
    1200
  );

  const list = Array.isArray(result.topics) ? result.topics : [];
  return list
    .filter((t) => t && t.topic)
    .slice(0, 3)
    .map((t) => ({
      topic: String(t.topic).trim(),
      area: String(t.area || "AI").trim(),
      reason: String(t.reason || "").trim(),
    }));
}

// ===================== STEP 2: DEEP RESEARCH =====================

async function fetchWebResults(topic) {
  const url =
    "https://serpapi.com/search.json?engine=google" +
    "&hl=en&gl=us&num=10" +
    "&q=" + encodeURIComponent(topic) +
    "&api_key=" + encodeURIComponent(SERPAPI_KEY);
  try {
    const data = await fetchJson(url, {}, "SerpAPI google web");
    const organic = Array.isArray(data.organic_results) ? data.organic_results : [];
    return organic
      .filter((r) => r && typeof r.title === "string" && r.link)
      .map((r) => ({
        title: r.title.trim(),
        link: r.link,
        source: r.source || new URL(r.link).hostname,
        date: r.date || "",
        snippet: r.snippet || "",
      }));
  } catch (err) {
    console.error("WARNING: web search skipped — " + err.message);
    return [];
  }
}

function stripHtml(html, maxChars) {
  let text = String(html)
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
  return text.slice(0, maxChars);
}

async function fetchArticleText(link) {
  try {
    const res = await fetch(link, {
      signal: AbortSignal.timeout(15000),
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        Accept: "text/html,application/xhtml+xml",
      },
    });
    if (!res.ok) return "";
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("html")) return "";
    const html = await res.text();
    return stripHtml(html, 1500);
  } catch {
    return "";
  }
}

async function researchTopic(topic, seedHeadlines) {
  console.log("Research phase: searching news + web for: " + topic);

  let newsItems = [];
  try {
    newsItems = await fetchNewsHeadlines(topic);
    console.log(`Research: ${newsItems.length} news results for the exact topic`);
  } catch (err) {
    console.error("WARNING: " + err.message);
  }

  const webItems = await fetchWebResults(topic);
  console.log(`Research: ${webItems.length} web results for the exact topic`);

  const seen = new Set();
  const merged = [];
  for (const item of [...newsItems, ...webItems]) {
    const key = item.link || item.title.toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    merged.push(item);
  }

  // Add seed headlines as weak sources too (no link text available).
  for (const h of seedHeadlines) {
    const key = h.title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push({ title: h.title, link: "", source: h.source || "news", date: h.date || "", snippet: h.snippet || "" });
  }

  if (merged.length === 0) {
    throw new Error("Research found zero sources for this topic.");
  }

  // Fetch the real article text of the top sources so facts come from actual pages.
  const topLinks = merged.filter((m) => m.link).slice(0, 8);
  console.log(`Research: fetching full text of ${topLinks.length} source pages...`);
  const sources = [];
  for (const item of merged.slice(0, 8)) {
    const base = { ...item };
    if (item.link) {
      const fetched = topLinks.find((t) => t.link === item.link);
      if (fetched) base.text = await fetchArticleText(item.link);
      else base.text = "";
    } else {
      base.text = "";
    }
    sources.push(base);
  }

  const withText = sources.filter((s) => s.text && s.text.length > 200).length;
  console.log(`Research done: ${sources.length} sources total, ${withText} with full article text.`);
  if (withText === 0 && sources.every((s) => !s.snippet)) {
    throw new Error("Not enough readable source material to write a factual post.");
  }
  return sources;
}

// ===================== STEP 3: VERIFIED FACT SHEET =====================

async function buildFactSheet(topic, area, sources) {
  const sourceBlocks = sources
    .slice(0, 8)
    .map((s, i) => {
      const textPart = s.text
        ? `\nTEXT: ${s.text}`
        : s.snippet
          ? `\nSNIPPET: ${s.snippet.slice(0, 300)}`
          : "";
      return `SOURCE ${i + 1}\nTITLE: ${s.title}\nPUBLISHER: ${s.source}\nURL: ${s.link || "(none)"}${textPart}`;
    })
    .join("\n\n");

  const prompt = `Extract EVERY concrete fact about "${topic}" (area: ${area}) from these REAL sources only.

${sourceBlocks}

Rules: use ONLY what is written above. Never invent numbers, dates, prices, percentages, benchmarks or quotes. Extract all specific details: product/model names, versions, features, companies, availability, what it does, who made it. Prefer facts from 2+ sources (mark verified_multi:true). Up to 25 facts. "sourceUrls" = exact URLs supporting the fact.
Reply ONLY JSON: {"facts":[{"fact":"precise sentence","sourceUrls":["https://..."],"verified_multi":true|false}]}
Return {"facts":[]} only if the sources contain no usable facts about the topic at all.`;

  const result = await groqJson(
    [
      {
        role: "system",
        content: "You extract only supported facts. Never invent. Reply with valid JSON only.",
      },
      { role: "user", content: prompt },
    ],
    3500
  );

  const facts = Array.isArray(result.facts) ? result.facts.filter((f) => f && f.fact) : [];
  console.log(`Fact sheet: ${facts.length} verified facts extracted.`);
  return facts;
}

// ===================== STEP 4: STRICT POST WRITING =====================

function socialLinksHtml() {
  const urls = String(AUTHOR_SOCIALS || "")
    .split(",")
    .map((u) => u.trim())
    .filter(Boolean);
  const parts = urls.map((url) => {
    let label = "Link";
    try {
      const host = new URL(url).hostname.replace(/^www\./, "");
      if (host.includes("instagram")) label = "Instagram";
      else if (host.includes("t.me") || host.includes("telegram")) label = "Telegram";
      else if (host.includes("x.com") || host.includes("twitter")) label = "X";
      else if (host.includes("linkedin")) label = "LinkedIn";
      else label = host;
    } catch {
      /* keep default label */
    }
    return `<a href="${url}" target="_blank" rel="me noopener" style="color:#0b57d0;text-decoration:none;margin-right:12px;">${label}</a>`;
  });
  return parts.join(" ");
}

function authorBoxHtml() {
  return `
<div style="border:1px solid #dadce0;border-radius:12px;padding:20px 24px;margin-top:32px;background:#fafafa;font-family:Arial,sans-serif;">
  <h3 style="margin:0 0 10px 0;font-size:18px;color:#202124;">About the Author</h3>
  <p style="margin:0 0 8px 0;font-size:15px;color:#202124;"><strong><a href="${PORTFOLIO_URL}" target="_blank" rel="me noopener" style="color:#0b57d0;text-decoration:none;">${AUTHOR_NAME}</a></strong></p>
  <p style="margin:0 0 12px 0;font-size:14px;color:#3c4043;line-height:1.6;">${AUTHOR_BIO}</p>
  <p style="margin:0;font-size:13px;">${socialLinksHtml()}</p>
</div>`.trim();
}

function uniqueSourceLinks(facts, sources) {
  const urlToTitle = new Map();
  for (const s of sources) {
    if (s.link) urlToTitle.set(s.link, s.title);
  }
  const chosen = [];
  const seen = new Set();
  for (const f of facts) {
    for (const url of f.sourceUrls || []) {
      if (!url || seen.has(url)) continue;
      seen.add(url);
      chosen.push({ url, title: urlToTitle.get(url) || url });
      if (chosen.length >= 4) return chosen;
    }
  }
  // Fallback: real article sources if fact URLs were empty.
  for (const s of sources) {
    if (s.link && !seen.has(s.link) && s.text && s.text.length > 200) {
      seen.add(s.link);
      chosen.push({ url: s.link, title: s.title });
      if (chosen.length >= 4) break;
    }
  }
  return chosen;
}

async function writePost(topic, area, facts, refs, mentionAuthor) {
  const factList = facts
    .map((f, i) => `${i + 1}. ${f.fact}${f.verified_multi ? " [confirmed by multiple sources]" : ""}`)
    .join("\n");
  const refList = refs.map((r, i) => `[R${i + 1}] ${r.title} — ${r.url}`).join("\n");

  const authorRules = mentionAuthor
    ? `IMPORTANT BRANDING RULE: Mention the author naturally EXACTLY ONCE inside the body using this exact anchor: <a href="${PORTFOLIO_URL}" target="_blank" rel="noopener">${AUTHOR_NAME}</a>, with ONE short factual line about the author's work taken ONLY from this bio: "${AUTHOR_BIO}". Invent nothing about the author. NEVER put the author name in the title.`
    : `IMPORTANT BRANDING RULE: Do NOT mention the author name anywhere in the title or body.`;

  const prompt = `You are writing a fully fact-checked Hinglish blog post (Hindi in Latin script + natural English tech terms, "aap" tone) for an AI and tech blog.

Topic: ${topic}
Area: ${area}
Main keyword for the title AND first paragraph: "${topic}"

VERIFIED FACT SHEET (the ONLY facts you may use — every claim in the post must come from this list):
${factList}

SOURCE REFERENCES (use these exact URLs in the References section, cite 3-4 of them as [R1]/[R2] style or natural inline links where relevant):
${refList}

HARD RULES (violating any = failed post):
1. Write ONLY facts from the fact sheet. If a number, date, price, percentage, version, benchmark or quote is NOT in the fact sheet, do NOT write it. No general knowledge filler, no "experts believe", no predictions.
2. Do not add your own opinions presented as facts. You MAY add explanatory sentences that help the reader understand something already stated in the facts (what it means, why it matters), as long as they introduce no new fact.
3. LENGTH IS A REQUIREMENT: the final post must be 650 to 800 words. Write exactly 5 sections (use <h2>). Each section must have 2 full paragraphs of 3-5 sentences each. The intro and conclusion must each be at least 3 sentences. Do NOT stop early.
4. Structure: catchy title (max ~70 chars, include main keyword, NO author name) → intro with the main keyword → 5 sections with <h2> headings and 2 paragraphs each → conclusion → then a final section <h2>References</h2> with a <ul> of 3-4 <li><a href="URL">source title</a></li> items using the reference URLs above.
5. Clean HTML only: <p>, <h2>, <h3>, <ul>, <li>, <strong>, <em>, <a>. NO <html>, <head>, <body>, <script>, <style>, <img>, <figure>, NO markdown.
${authorRules}

Return ONLY this JSON:
{"title": "...", "html": "...", "image_query": "2-4 word English stock photo query", "image_alt": "descriptive alt text including the main keyword", "labels": ["3 to 5 short SEO labels"]}`;

  const post = await groqJson(
    [
      {
        role: "system",
        content:
          "You are an extremely careful Hinglish tech blog writer. Every sentence must be backed by the provided fact sheet. You always reply with valid JSON only.",
      },
      { role: "user", content: prompt },
    ],
    4500
  );

  if (!post.title || !post.html) {
    throw new Error("Groq post response missing title or html.");
  }
  post.title = String(post.title).trim();
  post.html = String(post.html).trim();
  post.image_query = String(post.image_query || topic).trim();
  post.image_alt = String(post.image_alt || topic).trim();
  post.labels = Array.isArray(post.labels)
    ? post.labels.map((l) => String(l).trim()).filter(Boolean).slice(0, 5)
    : [topic];
  if (post.labels.length < 3) post.labels = [...post.labels, topic, area].slice(0, 5);
  return post;
}

function countWords(html) {
  return String(html)
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/gi, " ")
    .split(/\s+/)
    .filter(Boolean).length;
}

async function expandPost(post, facts, targetWords) {
  const factList = facts.map((f, i) => `${i + 1}. ${f.fact}`).join("\n");
  const prompt = `The Hinglish blog post below is too short (${countWords(post.html)} words). Expand it to ${targetWords} words using ONLY the verified facts listed. Add more explanation, context and natural transitions — but do NOT invent any new fact, number, quote or claim. Keep the same title and HTML-only format (<p>, <h2>, <h3>, <ul>, <li>, <strong>, <em>, <a>). Keep the References section at the end.

FACTS:
${factList}

CURRENT POST HTML:
${post.html}

Return ONLY JSON: {"html": "full expanded html"}`;
  const result = await groqJson(
    [
      {
        role: "system",
        content: "You expand Hinglish tech articles strictly within given facts. Reply with valid JSON only.",
      },
      { role: "user", content: prompt },
    ],
    4500
  );
  if (result.html && countWords(result.html) > countWords(post.html)) {
    post.html = String(result.html).trim();
    console.log(`Expanded post to ~${countWords(post.html)} words.`);
  }
  return post;
}

// ===================== STEP 5: FINAL FACT-CHECK PASS =====================

async function factCheckPost(post, facts) {
  const factList = facts.map((f, i) => `${i + 1}. ${f.fact}`).join("\n");
  const prompt = `You are a ruthless fact-checker for a blog. Below is a Hinglish blog post and the ONLY verified facts it was allowed to use.

VERIFIED FACTS:
${factList}

POST TITLE: ${post.title}

POST HTML:
${post.html}

Rules:
- Find every factual claim in the post (names, versions, dates, numbers, features, availability, pricing, quotes).
- A claim is OK only if it matches the verified fact list.
- UNSUPPORTED CLAIM → FIX, DON'T DELETE: rewrite the sentence so it only says what the facts support (or replace it with a sentence built from the facts). Only remove a sentence completely if it truly cannot be made accurate OR it is pure hype with zero information.
- Do NOT shrink the post. Keep the final post 650-800 words and keep all 5 <h2> sections plus the intro, conclusion and <h2>References</h2> section.
- Do not change the author branding rules: title must not contain an author name.

Return ONLY this JSON:
{"ok": true or false, "removed_claims": ["short description of each unsupported claim you fixed or removed, empty array if none"], "title": "final title (or same)", "html": "final cleaned html"}`;

  const result = await groqJson(
    [
      { role: "system", content: "You are a strict fact-checker. Reply with valid JSON only." },
      { role: "user", content: prompt },
    ],
    4500
  );

  const removed = Array.isArray(result.removed_claims) ? result.removed_claims : [];
  if (result.html && String(result.html).trim().length > 100) {
    post.html = String(result.html).trim();
  }
  if (result.title && String(result.title).trim()) {
    post.title = String(result.title).trim();
  }
  if (removed.length > 0) {
    console.log("Fact-check removed/fixed " + removed.length + " unsupported claim(s):");
    for (const c of removed) console.log("  - " + c);
  } else {
    console.log("Fact-check passed: no unsupported claims found.");
  }
  return post;
}

// ===================== STEP 6: IMAGE =====================

async function fetchPixabayImage(query) {
  const url =
    "https://pixabay.com/api/?key=" + encodeURIComponent(PIXABAY_API_KEY) +
    "&q=" + encodeURIComponent(query) +
    "&image_type=photo&orientation=horizontal&safesearch=true&per_page=5&order=popular";
  const data = await fetchJson(url, {}, "Pixabay");
  const hits = Array.isArray(data.hits) ? data.hits : [];
  if (hits.length === 0) {
    const fallback = await fetchJson(
      "https://pixabay.com/api/?key=" + encodeURIComponent(PIXABAY_API_KEY) +
        "&q=" + encodeURIComponent("technology computer") +
        "&image_type=photo&orientation=horizontal&safesearch=true&per_page=5",
      {},
      "Pixabay fallback"
    );
    if (!Array.isArray(fallback.hits) || fallback.hits.length === 0) {
      throw new Error("No Pixabay image found.");
    }
    return fallback.hits[Math.floor(Math.random() * fallback.hits.length)];
  }
  return hits[Math.floor(Math.random() * hits.length)];
}

function imageHtml(image, alt) {
  const src = image.webformatURL || image.largeImageURL;
  const credit = image.pageURL || "https://pixabay.com/";
  return (
    `<figure style="margin:0 0 24px 0;">` +
    `<img src="${src}" alt="${alt}" style="width:100%;height:auto;display:block;border-radius:8px;" />` +
    `<figcaption style="font-size:12px;color:#70757a;margin-top:6px;">Image via <a href="${credit}" target="_blank" rel="noopener">Pixabay</a></figcaption>` +
    `</figure>`
  );
}

// ===================== STEP 7: GOOGLE + BLOGGER =====================

async function getAccessToken() {
  const body = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    client_secret: GOOGLE_CLIENT_SECRET,
    refresh_token: GOOGLE_REFRESH_TOKEN,
    grant_type: "refresh_token",
  });
  try {
    const data = await fetchJson(
      "https://oauth2.googleapis.com/token",
      {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      },
      "Google OAuth"
    );
    if (!data.access_token) throw new Error("No access_token in Google response.");
    return data.access_token;
  } catch (err) {
    if (String(err.message).includes("invalid_grant")) {
      fail("Refresh token expired, generate a new one in OAuth Playground");
    }
    throw err;
  }
}

async function publishToBlogger(title, content, labels) {
  const accessToken = await getAccessToken();
  const isDraft = !PUBLISH_LIVE;
  const url =
    `https://www.googleapis.com/blogger/v3/blogs/${BLOGGER_BLOG_ID}/posts` +
    (isDraft ? "?isDraft=true" : "");
  return fetchJson(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title, content, labels }),
  }, "Blogger posts.insert");
}

// ===================== MAIN =====================

async function main() {
  console.log("=== Daily blog run (fact-checked mode) ===");
  console.log("Mode: " + (PUBLISH_LIVE ? "LIVE" : "DRAFT"));

  const keywords = readJson(KEYWORDS_PATH, []);
  if (!Array.isArray(keywords) || keywords.length < 2) {
    fail("config/keywords.json must contain at least 2 keywords.");
  }
  const published = readJson(PUBLISHED_PATH, []);
  const publishedTopics = published.map((p) => String(p.topic || "").toLowerCase());

  const MAX_ROUNDS = 3;
  let chosen = null; // { topic, area, reason, sources, facts, refs }
  let roundsAsked = 0;
  let nullRounds = 0;
  let errorRounds = 0;
  let candidateErrors = 0;

  for (let round = 1; round <= MAX_ROUNDS && !chosen; round++) {
    const [kw1, kw2] = shuffle(keywords).slice(0, 2);
    console.log(`--- Discovery round ${round}/${MAX_ROUNDS} — keywords: ${kw1}, ${kw2}`);

    const headlinesByKeyword = {};
    for (const kw of [kw1, kw2]) {
      try {
        const items = await fetchNewsHeadlines(kw);
        headlinesByKeyword[kw] = items;
        console.log(`SerpAPI: ${items.length} headlines for "${kw}"`);
      } catch (err) {
        console.error("WARNING: " + err.message);
        headlinesByKeyword[kw] = [];
      }
    }

    const flat = Object.values(headlinesByKeyword).flat();
    if (flat.length === 0) {
      console.log(`Round ${round}: no headlines collected from SerpAPI.`);
      continue;
    }

    roundsAsked++;
    let candidates;
    try {
      candidates = await chooseTopics(headlinesByKeyword, publishedTopics);
    } catch (err) {
      errorRounds++;
      console.error(`Round ${round}: Groq error during topic selection — ${err.message}`);
      continue;
    }

    if (!candidates || candidates.length === 0) {
      nullRounds++;
      console.log(`Round ${round}: Groq found no topic that fits.`);
      continue;
    }
    console.log(
      `Round ${round}: ${candidates.length} candidate topic(s): ${candidates.map((c) => c.topic).join(" | ")}`
    );

    // Research each candidate until one has enough verified facts.
    for (const cand of candidates) {
      const topicLower = cand.topic.toLowerCase();
      const dup = published.some((p) => {
        const t = String(p.topic || "").toLowerCase();
        return t === topicLower || topicLower.includes(t) || t.includes(topicLower);
      });
      if (dup) {
        console.log(`Candidate "${cand.topic}" already published — skipping.`);
        continue;
      }

      let sources;
      try {
        sources = await researchTopic(cand.topic, flat);
      } catch (err) {
        console.error(`Candidate "${cand.topic}" research failed — ${err.message}`);
        candidateErrors++;
        continue;
      }

      let facts;
      try {
        facts = await buildFactSheet(cand.topic, cand.area, sources);
      } catch (err) {
        console.error(`Candidate "${cand.topic}" fact extraction failed — ${err.message}`);
        candidateErrors++;
        continue;
      }

      const refs = uniqueSourceLinks(facts, sources);
      if (facts.length >= 8 && refs.length > 0) {
        chosen = { ...cand, sources, facts, refs };
        break;
      }
      console.log(
        `Candidate "${cand.topic}" has only ${facts.length} fact(s) and ${refs.length} source(s) — trying next candidate.`
      );
    }
  }

  if (!chosen) {
    if (errorRounds > 0 || candidateErrors > 0) {
      fail(
        "Topic selection/research failed because of a Groq or data error, not because the niche was empty. " +
          `Rounds asked: ${roundsAsked}, empty: ${nullRounds}, Groq errors: ${errorRounds}, candidate errors: ${candidateErrors}. Check Groq/SerpAPI and re-run.`
      );
    }
    if (roundsAsked > 0 && nullRounds === roundsAsked) {
      console.log(
        "No topic clearly fits the niche (Groq found no fitting topic in every round). Exiting without publishing."
      );
      process.exit(0);
    }
    console.log("No candidate had enough verifiable facts. Exiting without publishing (no fake content added).");
    process.exit(0);
  }

  const sources = chosen.sources;
  const facts = chosen.facts;
  const refs = chosen.refs;
  console.log(`Topic chosen: "${chosen.topic}" (${chosen.area}) — ${chosen.reason}`);
  console.log(`Sources: ${sources.length} | Verified facts: ${facts.length} | References: ${refs.length}`);

  const mentionAuthor = Math.random() < 1 / 3;
  console.log("Author mention in body: " + (mentionAuthor ? "YES (branding post)" : "no"));

  // ---- Write post strictly from fact sheet ----
  let post;
  try {
    post = await writePost(chosen.topic, chosen.area, facts, refs, mentionAuthor);
  } catch (err) {
    fail(err.message);
  }
  console.log("Draft post written: " + post.title + " (~" + countWords(post.html) + " words)");

  // ---- Fact-check pass first (fix unsupported claims, keep length) ----
  try {
    post = await factCheckPost(post, facts);
  } catch (err) {
    console.error("WARNING: fact-check pass failed — " + err.message + " (keeping writer output)");
  }
  console.log("After fact-check: ~" + countWords(post.html) + " words");

  // ---- Only if still short, expand strictly within the fact sheet ----
  if (countWords(post.html) < 600) {
    console.log("Word count below 600 — expanding with fact sheet only...");
    try {
      post = await expandPost(post, facts, 700);
    } catch (err) {
      console.error("WARNING: expansion failed — " + err.message);
    }
  }

  // Hard stop: if still under 600 words, don't publish an incomplete post.
  const finalWords = countWords(post.html);
  console.log("Final word count: " + finalWords);
  if (finalWords < 600) {
    console.log("Post too short after all passes. Exiting without publishing.");
    process.exit(0);
  }

  // ---- Image ----
  let image;
  try {
    image = await fetchPixabayImage(post.image_query);
  } catch (err) {
    console.error("WARNING: " + err.message + " — publishing without image.");
    image = null;
  }

  let html = "";
  if (image) html += imageHtml(image, post.image_alt);
  html += post.html;
  html += "\n" + authorBoxHtml();

  // Safety: never let the author name leak into the title.
  const namePattern = new RegExp(AUTHOR_NAME.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  const finalTitle = post.title.replace(namePattern, "").replace(/\s{2,}/g, " ").trim();

  const labels = [...post.labels];
  if (mentionAuthor && !labels.some((l) => l.toLowerCase() === AUTHOR_NAME.toLowerCase())) {
    labels.push(AUTHOR_NAME);
  }

  let result;
  try {
    result = await publishToBlogger(finalTitle, html, labels);
  } catch (err) {
    fail(err.message);
  }

  published.push({
    topic: chosen.topic,
    area: chosen.area,
    title: finalTitle,
    id: result.id,
    url: result.url || "",
    status: PUBLISH_LIVE ? "live" : "draft",
    factCount: facts.length,
    sourceCount: refs.length,
    date: new Date().toISOString(),
  });
  writeJson(PUBLISHED_PATH, published);

  console.log("=== SUCCESS ===");
  console.log("Saved as: " + (PUBLISH_LIVE ? "live post" : "DRAFT"));
  console.log("Title: " + finalTitle);
  console.log("Facts used: " + facts.length + " | Sources in refs: " + refs.length);
  console.log("Post URL: " + (result.url || ("(draft id " + result.id + ")")));
  console.log("published.json updated.");
}

main().catch((err) => {
  fail("Unexpected failure: " + err.message);
});
