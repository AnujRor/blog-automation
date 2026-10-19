import "dotenv/config";

const REQUIRED = [
  "GOOGLE_CLIENT_ID",
  "GOOGLE_CLIENT_SECRET",
  "GOOGLE_REFRESH_TOKEN",
  "BLOGGER_BLOG_ID",
];

const missing = REQUIRED.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error(`Google check failed: missing env variable(s): ${missing.join(", ")}.`);
  console.error("Simple baat: .env (ya GitHub Secrets) mein ye Google variables daal do.");
  process.exit(1);
}

const {
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
  GOOGLE_REFRESH_TOKEN,
  BLOGGER_BLOG_ID,
} = process.env;

const HINGLISH = {
  invalid_grant: "Refresh token expire ya revoke ho gaya hai — OAuth Playground se naya refresh token banao.",
  invalid_client: "Client ID ya Client Secret galat hai — .env / GitHub Secrets check karo.",
  invalid_request: "Request adhoori hai — Google variables sahi format mein check karo.",
  access_denied: "Google ne access deny kar diya — OAuth Playground mein sahi account aur scope se dobara authorize karo.",
  unauthorized_client:
    "Ye client refresh token use nahi kar sakta — OAuth Playground wahi Client ID/Secret use karta hai ya nahi, check karo.",
  insufficientPermissions: "Blogger API ki permission nahi hai — OAuth consent mein Blogger scope add karo.",
};

function explain(code) {
  if (HINGLISH[code]) return HINGLISH[code];
  return "Google ne error diya hai — settings aur OAuth setup check karo.";
}

async function main() {
  // --- Step 1: refresh token -> access token ---
  let tokenData;
  try {
    const res = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        refresh_token: GOOGLE_REFRESH_TOKEN,
        grant_type: "refresh_token",
      }).toString(),
    });
    tokenData = await res.json();
    if (!res.ok || !tokenData.access_token) {
      const code = tokenData.error || "unknown_error";
      console.error(`Google login failed: ${code}`);
      console.error(explain(code));
      process.exit(1);
    }
  } catch (err) {
    console.error("Google login failed: network_error");
    console.error("Simple baat: internet ya Google server issue hai — dobara try karo.");
    process.exit(1);
  }

  // --- Step 2: Blogger blogs.get ---
  let blogData;
  try {
    const res = await fetch(
      `https://www.googleapis.com/blogger/v3/blogs/${BLOGGER_BLOG_ID}`,
      { headers: { Authorization: `Bearer ${tokenData.access_token}` } }
    );
    blogData = await res.json();
    if (!res.ok) {
      const code = blogData?.error?.status || blogData?.error?.code || "api_error";
      const message = blogData?.error?.message || "";
      console.error(`Google Blogger check failed: ${code}`);
      if (/invalid_grant/i.test(message)) {
        console.error(explain("invalid_grant"));
      } else if (res.status === 403 || res.status === 404) {
        console.error(
          "Simple baat: BLOGGER_BLOG_ID galat hai ya us blog ki permission nahi hai — Blog ID aur account check karo."
        );
      } else {
        console.error(explain(code));
      }
      process.exit(1);
    }
  } catch (err) {
    console.error("Google Blogger check failed: network_error");
    console.error("Simple baat: Blogger API tak nahi pahunch paaye — internet ya Google server issue hai.");
    process.exit(1);
  }

  console.log(`Blog name: ${blogData.name || "(no name)"}`);
  console.log("GOOGLE OK");
}

main().catch(() => {
  console.error("Google login failed: unexpected_error");
  console.error("Simple baat: kuch unexpected hua — dobara try karo.");
  process.exit(1);
});
