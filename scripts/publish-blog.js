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
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "");
  const first = cleaned.indexOf("{");
  const last = cleaned.lastIndexOf("}");
  const firstArr = cleaned.indexOf("[");
  const lastArr = cleaned.lastIndexOf("]");
  // Prefer whichever JSON structure appears outermost and valid.
  if (first !== -1 && last !== -1 && last > first) {
    if (firstArr !== -1 && lastArr !== -1 && firstArr < first && lastArr > last) {
      return JSON.parse(cleaned.slice(firstArr, lastArr + 1));
    }
    return JSON.parse(cleaned.slice(first, last + 1));
  }
  if (firstArr !== -1 && lastArr !== -1 && lastArr > firstArr) {
    return JSON.parse(cleaned.slice(firstArr, lastArr + 1));
  }
  throw new Error("No JSON found in model response.");
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

let cachedModelId = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Groq free tier limits input tokens per minute (ITPM). We keep a rolling
// 60s window and wait when a call would cross the safe budget.
const TOKEN_WINDOW_MS = 60000;
const SAFE_ITPM = 6000;
let tokenWindow = [];

async function throttleInput(inputTokens) {
  tokenWindow = tokenWindow.filter((t) => Date.now() - t.time < TOKEN_WINDOW_MS);
  const used = tokenWindow.reduce((a, t) => a + t.tokens, 0);
  if (used + inputTokens > SAFE_ITPM && tokenWindow.length > 0) {
    const oldest = tokenWindow[0];
    const waitMs = TOKEN_WINDOW_MS - (Date.now() - oldest.time) + 800;
    console.log(
      "Rate limit: waiting " + Math.ceil(waitMs / 1000) + "s to respect Groq tokens/min limit..."
    );
    await sleep(waitMs);
    return throttleInput(inputTokens);
  }
  tokenWindow.push({ time: Date.now(), tokens: inputTokens });
}

async function pickGroqChatModel() {
  if (cachedModelId) return cachedModelId;
  const data = await fetchJson(
    "https://api.groq.com/openai/v1/models",
    { headers: { Authorization: `Bearer ${GROQ_API_KEY}` } },
    "Groq model list"
  );
  const models = Array.isArray(data.data) ? data.data : [];
  const ids = models
    .map((m) => m.id)
    .filter((id) => typeof id === "string" && id.length > 0);
  if (ids.length === 0) {
    throw new Error("No models available on your Groq account.");
  }
  const isChatLike = (id) =>
    !/whisper|guard|tts|stt|audio|orpheus|embedding|rerank/i.test(id);
  const llamaIds = ids.filter((id) => id.toLowerCase().includes("llama") && isChatLike(id));
  const preferred =
    llamaIds.find((id) => id.includes("llama-3.3-70b")) ||
    llamaIds.find((id) => id.includes("versatile")) ||
    llamaIds[0] ||
    ids.filter(isChatLike).find((id) => id.includes("gpt-oss-120b")) ||
    ids.filter(isChatLike).find((id) => id.includes("qwen")) ||
    ids.filter(isChatLike)[0];
  if (!preferred) {
    throw new Error("No usable chat model available on your Groq account.");
  }
  cachedModelId = preferred;
  console.log("Using Groq model: " + preferred);
  return preferred;
}

async function groqChat(messages, maxTokens = 2000) {
  const model = await pickGroqChatModel();
  const inputChars = messages.reduce((a, m) => a + String(m.content || "").length, 0);
  const inputTokens = Math.ceil(inputChars / 4) + 50;

  for (let attempt = 1; attempt <= 6; attempt++) {
    await throttleInput(inputTokens + maxTokens);
    let res;
    try {
      res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.4,
          max_tokens: maxTokens,
          response_format: { type: "json_object" },
        }),
      });
    } catch (err) {
      throw new Error("Groq network error: " + err.message);
    }

    const text = await res.text();

    if (res.status === 429) {
      const match = text.match(/try again in ([\d.]+)s/i) || text.match(/in ([\d.]+)s/i);
      const headerRetry = parseFloat(res.headers.get("retry-after") || "0");
      const waitMs = Math.max(
        match ? parseFloat(match[1]) * 1000 : 0,
        headerRetry ? headerRetry * 1000 : 0,
        15000
      );
      console.log(
        `Groq rate limit (attempt ${attempt}) — waiting ${Math.ceil(waitMs / 1000)}s and retrying...`
      );
      await sleep(waitMs + 1000);
      continue;
    }

    if (res.status === 400 && /json/i.test(text)) {
      console.log(`Groq returned invalid JSON (attempt ${attempt}) — retrying...`);
      await sleep(2500);
      continue;
    }

    if (!res.ok) {
      let detail = text.slice(0, 300);
      try {
        const parsed = JSON.parse(text);
        detail = parsed.error?.message || parsed.error || detail;
      } catch {
        /* keep raw text */
      }
      throw new Error(`Groq chat failed (HTTP ${res.status}): ${detail}`);
    }

    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("Groq returned non-JSON response.");
    }
    const content = data?.choices?.[0]?.message?.content;
    if (!content) throw new Error("Groq returned an empty response.");
    return content;
  }

  throw new Error("Groq rate limit persisted after multiple retries. Try again in a minute.");
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

async function chooseTopic(headlinesByKeyword, publishedTopics) {
  const flat = [];
  for (const [keyword, items] of Object.entries(headlinesByKeyword)) {
    for (const it of items) flat.push(`[${keyword}] ${it.title} (${it.source}${it.date ? ", " + it.date : ""})`);
  }
  if (flat.length === 0) return null;

  const prompt = `You are the strict topic selector for a Hinglish blog that ONLY covers these three areas:
1) AI: new AI tools, AI models, chatbots, AI news
2) Tech: AI-assisted coding, web development, programming, automation and APIs, new software or tech products
3) Digital marketing with AI: AI tools for SEO, social media, content marketing, ads, email marketing, growth

FORBIDDEN: gaming, politics, celebrity, sports, crime, health, finance tips, general entertainment, and any digital marketing topic with no AI/tech tool angle. Reject anything unclear.

Today's real news headlines:
${flat.join("\n")}

Topics already published before (never pick these or near-duplicates):
${publishedTopics.length ? publishedTopics.join("\n") : "(none yet)"}

Pick the SINGLE best topic that: clearly fits one of the three areas, is NOT already published, has MULTIPLE headlines here (so it can be cross-verified from several sources), and is concrete (a specific tool/model/feature/release — not a vague theme).

Return ONLY this JSON:
{"topic": "short precise English topic phrase", "area": "AI" or "Tech" or "Digital marketing with AI", "reason": "one sentence"}

If nothing clearly fits, return {"topic": null, "area": null, "reason": null}.`;

  const text = await groqChat([
    {
      role: "system",
      content:
        "You are a strict topic selector for an AI/tech Hinglish blog. Reply only with valid JSON.",
    },
    { role: "user", content: prompt },
  ], 500);

  const result = extractJson(text);
  if (!result.topic) return null;
  return {
    topic: String(result.topic).trim(),
    area: String(result.area || "AI").trim(),
    reason: String(result.reason || "").trim(),
  };
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
    return stripHtml(html, 1600);
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
  const topLinks = merged.filter((m) => m.link).slice(0, 5);
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
    .map((s, i) => {
      const textPart = s.text
        ? `\nARTICLE TEXT (truncated): ${s.text}`
        : s.snippet
          ? `\nSNIPPET: ${s.snippet}`
          : "\n(no extra text available)";
      return `SOURCE ${i + 1}
TITLE: ${s.title}
PUBLISHER: ${s.source}
DATE: ${s.date || "unknown"}
URL: ${s.link || "(headline only, no URL)"}${textPart}`;
    })
    .join("\n\n");

  const prompt = `You are a meticulous fact-checker. Below are REAL sources about the topic: "${topic}" (area: ${area}).

${sourceBlocks}

Task: Extract ONLY facts that are clearly supported by the source text above. Rules:
- Every fact must be verifiable from at least one source listed. NO guesses, NO common-knowledge filler, NO invented numbers, prices, dates, percentages, benchmark scores, user counts or quotes.
- Prefer facts stated by 2+ sources (mark those "verified_multi": true).
- Include concrete details only: exact feature names, model names, version numbers, company names, dates, what it does, availability/pricing ONLY if a source states it.
- Maximum 20 facts. Quality over quantity.
- "sourceUrls": the exact URLs of the sources that support this fact (empty array only for headline-only items).

Return ONLY this JSON:
{"facts": [{"fact": "one precise sentence", "sourceUrls": ["https://..."], "verified_multi": true or false}]}

If the sources do not contain enough solid facts (fewer than 5), return {"facts": []}.`;

  const text = await groqChat([
    {
      role: "system",
      content:
        "You are a strict fact-extraction engine. You never invent anything. Reply only with valid JSON.",
    },
    { role: "user", content: prompt },
  ], 3000);

  const result = extractJson(text);
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
2. Do not add your own opinions presented as facts.
3. Structure: catchy title (max ~70 chars, include main keyword, NO author name) → short intro with the main keyword → 4 to 5 sections with clean <h2>/<h3> headings and 1-2 paragraphs each → conclusion → then a final section <h2>References</h2> with a <ul> of 3-4 <li><a href="URL">source title</a></li> items using the reference URLs above.
4. Total body length INCLUDING headings and References: 600 to 800 words.
5. Clean HTML only: <p>, <h2>, <h3>, <ul>, <li>, <strong>, <em>, <a>. NO <html>, <head>, <body>, <script>, <style>, <img>, <figure>, NO markdown.
${authorRules}

Return ONLY this JSON:
{"title": "...", "html": "...", "image_query": "2-4 word English stock photo query", "image_alt": "descriptive alt text including the main keyword", "labels": ["3 to 5 short SEO labels"]}`;

  const text = await groqChat([
    {
      role: "system",
      content:
        "You are an extremely careful Hinglish tech blog writer. Every sentence must be backed by the provided fact sheet. You always reply with valid JSON only.",
    },
    { role: "user", content: prompt },
  ], 4500);

  const post = extractJson(text);
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
  const text = await groqChat([
    { role: "system", content: "You expand Hinglish tech articles strictly within given facts. Reply with valid JSON only." },
    { role: "user", content: prompt },
  ], 4500);
  const result = extractJson(text);
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
- Remove or rewrite any unsupported claim so the post stays accurate and natural. Also remove any sentence that adds no information (filler/hype with no fact).
- Do not change the author branding rules: title must not contain an author name.
- Keep 600-800 words if possible. Keep the References <h2> section at the end.

Return ONLY this JSON:
{"ok": true or false, "removed_claims": ["short description of each unsupported claim removed, empty array if none"], "title": "final title (or same)", "html": "final cleaned html"}`;

  const text = await groqChat([
    { role: "system", content: "You are a strict fact-checker. Reply with valid JSON only." },
    { role: "user", content: prompt },
  ], 4500);

  const result = extractJson(text);
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

  const [kw1, kw2] = shuffle(keywords).slice(0, 2);
  console.log("Random keywords picked: " + [kw1, kw2].join(", "));

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

  const seedHeadlines = Object.values(headlinesByKeyword).flat();
  if (seedHeadlines.length === 0) {
    console.log("No headlines collected from SerpAPI. Exiting without publishing.");
    process.exit(0);
  }

  const chosen = await chooseTopic(headlinesByKeyword, publishedTopics);
  if (!chosen || !chosen.topic) {
    console.log("No topic clearly fits the niche. Exiting without publishing.");
    process.exit(0);
  }
  console.log(`Topic chosen: "${chosen.topic}" (${chosen.area}) — ${chosen.reason}`);

  const topicLower = chosen.topic.toLowerCase();
  const alreadyPublished = published.some((p) => {
    const t = String(p.topic || "").toLowerCase();
    return t === topicLower || topicLower.includes(t) || t.includes(topicLower);
  });
  if (alreadyPublished) {
    console.log("Topic already in published.json. Exiting without publishing.");
    process.exit(0);
  }

  // ---- Deep research on the chosen topic ----
  let sources;
  try {
    sources = await researchTopic(chosen.topic, seedHeadlines);
  } catch (err) {
    fail("Research failed: " + err.message);
  }

  // ---- Verified fact sheet ----
  let facts = [];
  try {
    facts = await buildFactSheet(chosen.topic, chosen.area, sources);
  } catch (err) {
    fail("Fact extraction failed: " + err.message);
  }
  if (facts.length < 5) {
    console.log(
      `Only ${facts.length} verified facts found — not enough for a trustworthy post. Exiting without publishing.`
    );
    process.exit(0);
  }

  const refs = uniqueSourceLinks(facts, sources);
  if (refs.length === 0) {
    console.log("No real source URLs available for a References section. Exiting without publishing.");
    process.exit(0);
  }
  console.log(`Using ${refs.length} real source links in References.`);

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

  // ---- Word count enforcement (one expansion retry) ----
  if (countWords(post.html) < 600) {
    console.log("Word count below 600 — expanding with fact sheet only...");
    try {
      post = await expandPost(post, facts, 700);
    } catch (err) {
      console.error("WARNING: expansion failed — " + err.message);
    }
  }

  // ---- Final fact-check pass ----
  try {
    post = await factCheckPost(post, facts);
  } catch (err) {
    console.error("WARNING: fact-check pass failed — " + err.message + " (keeping writer output)");
  }

  // Hard stop: if still under 500 words, don't publish junk.
  const finalWords = countWords(post.html);
  console.log("Final word count: " + finalWords);
  if (finalWords < 500) {
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
