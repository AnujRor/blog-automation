import "dotenv/config";

const UNSPLASH_ACCESS_KEY = process.env.UNSPLASH_ACCESS_KEY;
const PIXABAY_API_KEY = process.env.PIXABAY_API_KEY;

const QUERY = "artificial intelligence";

function mask(value) {
  return value ? "set (hidden)" : "MISSING";
}

async function getJson(url, headers) {
  const res = await fetch(url, { headers });
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }
  return { ok: res.ok, status: res.status, data, text };
}

async function testUnsplash() {
  console.log("=== Unsplash test (query: \"" + QUERY + "\") ===");
  console.log("UNSPLASH_ACCESS_KEY: " + mask(UNSPLASH_ACCESS_KEY));
  if (!UNSPLASH_ACCESS_KEY) {
    console.log("RESULT: SKIPPED — Unsplash key missing.");
    return false;
  }

  const url =
    "https://api.unsplash.com/search/photos?query=" + encodeURIComponent(QUERY) +
    "&orientation=landscape&per_page=8&content_filter=high";
  let out;
  try {
    out = await getJson(url, { Authorization: "Client-ID " + UNSPLASH_ACCESS_KEY });
  } catch (err) {
    console.log("RESULT: FAILED — network error: " + err.message);
    return false;
  }

  if (!out.ok) {
    const reason =
      out.data && (out.data.errors || out.data.error)
        ? JSON.stringify(out.data.errors || out.data.error)
        : "HTTP " + out.status;
    console.log("RESULT: FAILED — " + reason);
    return false;
  }

  const results = out.data && Array.isArray(out.data.results) ? out.data.results : [];
  if (results.length === 0) {
    console.log("RESULT: FAILED — no photos found.");
    return false;
  }

  const photo = results[0];
  console.log("Photos found: " + results.length);
  console.log("Photographer: " + ((photo.user && photo.user.name) || "Unknown"));
  console.log("Image URL (regular): " + (photo.urls && photo.urls.regular));
  console.log("Unsplash profile: " + (photo.user && photo.user.links && photo.user.links.html));

  const dl = photo.links && photo.links.download_location;
  let downloadOk = false;
  if (!dl) {
    console.log("download_location: MISSING on this photo");
  } else {
    try {
      const res = await fetch(dl, {
        headers: { Authorization: "Client-ID " + UNSPLASH_ACCESS_KEY },
      });
      downloadOk = res.ok;
      console.log("download_location call: " + (res.ok ? "OK" : "FAILED (HTTP " + res.status + ")"));
    } catch (err) {
      console.log("download_location call: FAILED — " + err.message);
    }
  }

  if (downloadOk) console.log("RESULT: OK");
  return downloadOk;
}

async function testPixabay() {
  console.log("");
  console.log("=== Pixabay fallback test (query: \"" + QUERY + "\") ===");
  console.log("PIXABAY_API_KEY: " + mask(PIXABAY_API_KEY));
  if (!PIXABAY_API_KEY) {
    console.log("RESULT: SKIPPED — Pixabay key missing.");
    return false;
  }

  const url =
    "https://pixabay.com/api/?key=" + encodeURIComponent(PIXABAY_API_KEY) +
    "&q=" + encodeURIComponent(QUERY) +
    "&image_type=photo&orientation=horizontal&safesearch=true&per_page=5&order=popular";
  let out;
  try {
    out = await getJson(url);
  } catch (err) {
    console.log("RESULT: FAILED — network error: " + err.message);
    return false;
  }

  if (!out.ok) {
    console.log("RESULT: FAILED — HTTP " + out.status);
    return false;
  }

  const hits = out.data && Array.isArray(out.data.hits) ? out.data.hits : [];
  if (hits.length === 0) {
    console.log("RESULT: FAILED — no photos found.");
    return false;
  }

  const image = hits[0];
  console.log("Photos found: " + hits.length);
  console.log("Photographer: " + (image.user || "Unknown"));
  console.log("Image URL: " + (image.webformatURL || image.largeImageURL));
  console.log("Credit page: " + (image.pageURL || "https://pixabay.com/"));
  console.log("RESULT: OK");
  return true;
}

async function main() {
  console.log("Image source check — no Groq, no Blogger, no publishing.");
  console.log("");
  const unsplashOk = await testUnsplash();
  const pixabayOk = await testPixabay();
  console.log("");
  console.log("=== SUMMARY ===");
  console.log("Unsplash: " + (unsplashOk ? "OK" : "NOT OK"));
  console.log("Pixabay fallback: " + (pixabayOk ? "OK" : "NOT OK"));
  if (!unsplashOk && !pixabayOk) {
    console.log("Both image sources failed. Posts will publish without images.");
    process.exit(1);
  }
  console.log("At least one image source works. Publishing can continue.");
}

main().catch((err) => {
  console.error("Unexpected failure: " + err.message);
  process.exit(1);
});
