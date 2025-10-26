# Supabase Setup Guide

## ✅ What's Already Done

1. ✅ Python dependencies installed (PyJWT, httpx)
2. ✅ Next.js dependencies installed (@supabase/supabase-js, @supabase/ssr)
3. ✅ All code files created
4. ✅ Environment variables template added to `.env`

## 📋 What You Need To Do

### Step 1: Create Supabase Project

1. Go to https://supabase.com and sign up/login
2. Click "New Project"
3. Fill in:
   - Name: `hackohio-project` (or whatever you want)
   - Database Password: **remember this!**
   - Region: Choose closest to you
4. Wait ~2 minutes for project to be created

### Step 2: Get Your Supabase Credentials

1. In your Supabase project dashboard
2. Go to **Settings** → **API**
3. Copy these values:
   - **Project URL** (looks like `https://xxxxx.supabase.co`)
   - **anon public** key (long string starting with `eyJ...`)

### Step 3: Update Your .env File

Open `.env` and replace the placeholder values:

```bash
# Replace these lines in your .env file:
SUPABASE_URL=https://YOUR-PROJECT-ID.supabase.co
NEXT_PUBLIC_SUPABASE_URL=https://YOUR-PROJECT-ID.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-actual-anon-key-here
```

### Step 4: Run the Database Setup SQL

1. In Supabase dashboard, go to **SQL Editor**
2. Click **New Query**
3. Copy the entire contents of `supabase_setup.sql`
4. Paste it into the SQL editor
5. Click **Run** (or press Cmd/Ctrl + Enter)
6. You should see "Success. No rows returned"

### Step 5: Test Everything

**Terminal 1 - Backend:**
```bash
python main.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

Then visit: http://localhost:3000/sign-in

### Step 6: Sign In with Google

1. Click "Continue with Google"
2. You'll be redirected to Google sign-in
3. After signing in, you'll be redirected back
4. You should land on the `/onboarding` page
5. Fill out the hair preferences survey
6. You'll land on `/discover` page
7. Click "Ping backend (JWT)" to test backend auth

## 🔧 Troubleshooting

### "Missing token" error
- Make sure you've updated the `.env` file with real Supabase credentials
- Restart both servers after updating `.env`

### "Supabase not configured" error
- Check that `SUPABASE_URL` is set in `.env`
- It should be `https://xxxxx.supabase.co` (not placeholder)

### Can't sign in with Google
- Go to Supabase dashboard → Authentication → Providers
- Enable Google provider
- Add your domain to allowed redirect URLs

### SQL errors when running setup
- Make sure you're running it in the SQL Editor (not trying to import as a file)
- The entire SQL should run in one go

## 📝 Quick Checklist

- [ ] Created Supabase project
- [ ] Copied Project URL and anon key
- [ ] Updated `.env` file with real values
- [ ] Ran `supabase_setup.sql` in SQL Editor
- [ ] Started backend server (`python main.py`)
- [ ] Started frontend server (`npm run dev`)
- [ ] Tested sign-in flow
- [ ] Completed onboarding survey
- [ ] Tested protected `/discover` page

## 🎉 You're Done!

Once everything is working, you have:
- ✅ Google + Magic link authentication
- ✅ Protected routes with middleware
- ✅ JWT verification on backend
- ✅ User preferences stored in Supabase
- ✅ Row-level security enabled

