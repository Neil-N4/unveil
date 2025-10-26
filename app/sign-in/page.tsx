'use client'
import { useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { createClient } from '@/lib/supabase'

export default function SignIn() {
  const supabase = createClient()
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const next = useSearchParams().get('next') || '/post-auth'
  const redirectTo = typeof window !== 'undefined'
    ? `${location.origin}/auth/callback?next=${encodeURIComponent(next)}`
    : undefined

  async function signInGoogle() {
    await supabase.auth.signInWithOAuth({ provider: 'google', options: { redirectTo } })
  }
  async function sendMagic() {
    const { error } = await supabase.auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo } })
    if (!error) setSent(true)
  }

  return (
    <div className="mx-auto max-w-sm p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Sign in</h1>
      <button onClick={signInGoogle} className="w-full rounded-xl bg-black text-white py-2">Continue with Google</button>
      <div className="text-center text-sm text-white/60">or</div>
      {sent ? <p className="text-emerald-400 text-sm">Magic link sent. Check your email.</p> : (
        <>
          <input className="w-full rounded-xl bg-white/5 border border-white/10 px-3 py-2"
                 placeholder="you@site.com" value={email} onChange={e=>setEmail(e.target.value)} />
          <button onClick={sendMagic} className="w-full rounded-xl bg-indigo-600 text-white py-2">Send magic link</button>
        </>
      )}
    </div>
  )
}
