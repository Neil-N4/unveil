# Supabase Auth Integration Setup

## ✅ What's Been Created

### Frontend (Next.js)
- ✅ `lib/supabase.ts` - Supabase client helpers
- ✅ `lib/api.ts` - Auth-aware fetch helper
- ✅ `app/sign-in/page.tsx` - Google + Magic link sign-in
- ✅ `app/auth/callback/route.ts` - OAuth callback handler
- ✅ `app/post-auth/page.tsx` - Routes to onboarding or discover
- ✅ `app/onboarding/page.tsx` - First-time user survey
- ✅ `app/discover/page.tsx` - Protected app page
- ✅ `middleware.ts` - Protects `/discover` and `/results` routes

### Backend (FastAPI)
- ✅ JWT verification functions added to `main.py`
- ✅ `/api/ping` endpoint (protected, requires JWT)
- ✅ `/api/search` endpoint (protected, requires JWT)

### Database
- ✅ `supabase_setup.sql` - Ready to run in Supabase SQL Editor

## 📋 Setup Steps

### 1. Install Python dependencies
```bash
pip install PyJWT httpx
```

### 2. Supabase Setup

1. Go to https://supabase.com → Create project
2. Copy your Project URL and anon key from Settings → API
3. Run the SQL in `supabase_setup.sql`:
   - Go to SQL Editor in Supabase
   - Paste and run the entire contents of `supabase_setup.sql`

### 3. Environment Variables

Add to your `.env` file:
```bash
# Supabase (add these)
SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here

# Backend API
NEXT_PUBLIC_API_BASE=http://localhost:8083

# Reddit API (already have these)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...

# Mistral AI (if using)
MISTRAL_API_KEY=...
```

### 4. Install Next.js dependencies
```bash
npm install @supabase/supabase-js @supabase/ssr
```

### 5. Run Both Servers

**Terminal 1 - Backend:**
```bash
python main.py
# Server runs on http://localhost:8083
```

**Terminal 2 - Frontend:**
```bash
npm run dev
# Server runs on http://localhost:3000
```

### 6. Test the Flow

1. Visit http://localhost:3000/sign-in
2. Sign in with Google (or magic link)
3. You'll be redirected to `/post-auth` → then `/onboarding`
4. Fill out the hair survey
5. Land on `/discover`
6. Click "Ping backend (JWT)" - should show your user_id

## 🔒 How It Works

1. **User signs in** → Supabase creates JWT token
2. **Middleware checks** for auth cookies on protected routes
3. **Frontend sends JWT** in Authorization header to backend
4. **Backend verifies JWT** using Supabase public keys
5. **User data accessible** via `user.get("sub")` (user_id)

## 📝 Notes

- The `/api/search` endpoint now requires authentication
- Public endpoints like `/` and `/search` still work without auth
- User preferences stored in `beauty_preferences` table
- Row-level security ensures users only see their own data

