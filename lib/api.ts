import { createClient } from './supabase'

export async function authFetch(url: string, options: RequestInit = {}) {
  const supabase = createClient()
  const { data: { session } } = await supabase.auth.getSession()
  const token = session?.access_token
  return fetch(url, {
    ...options,
    headers: { ...(options.headers||{}), ...(token ? { Authorization: `Bearer ${token}` } : {}) }
  })
}

