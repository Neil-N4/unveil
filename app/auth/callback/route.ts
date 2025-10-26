import { NextRequest, NextResponse } from 'next/server'
import { createServerClient, type CookieOptions } from '@supabase/ssr'

export async function GET(req: NextRequest) {
  const url = req.nextUrl
  const code = url.searchParams.get('code')
  const nextPath = url.searchParams.get('next') || '/post-auth'
  const origin = url.origin

  const redirectToSignIn = (message?: string) =>
    NextResponse.redirect(new URL(
      message ? `/sign-in?error=${encodeURIComponent(message)}` : '/sign-in',
      origin
    ))

  const response = NextResponse.redirect(new URL(nextPath, origin))

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return req.cookies.get(name)?.value
        },
        set(name: string, value: string, options: CookieOptions) {
          response.cookies.set({ name, value, ...options })
        },
        remove(name: string, options: CookieOptions) {
          response.cookies.set({ name, value: '', ...options })
        },
      },
    }
  )

  if (!code) {
    return redirectToSignIn('Missing verification code')
  }

  const { error } = await supabase.auth.exchangeCodeForSession(code)
  if (error) {
    return redirectToSignIn(error.message)
  }

  return response
}
