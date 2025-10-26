'use client'
import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { createClient } from '@/lib/supabase'
import { authFetch } from '@/lib/api'

type BeautyPrefs = {
  hair_type: string | null
  concerns: string[] | null
  price_range: string | null
}

type SearchResult = {
  name: string
  summary?: string
  score?: number
  pros?: string[]
  cons?: string[]
  urls?: string[]
}

export default function Discover() {
  const supabase = useMemo(() => createClient(), [])
  const [email, setEmail] = useState('')
  const [prefs, setPrefs] = useState<BeautyPrefs | null>(null)
  const [searchInput, setSearchInput] = useState('')
  const [loadingPrefs, setLoadingPrefs] = useState(true)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])

  useEffect(() => {
    let active = true
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user || !active) return
      setEmail(user.email ?? '')
      const { data } = await supabase
        .from('beauty_preferences')
        .select('hair_type, concerns, price_range')
        .eq('user_id', user.id)
        .maybeSingle()
      if (active) {
        setPrefs(data ?? null)
        setSearchInput(buildSuggestedQuery(data))
        setLoadingPrefs(false)
      }
    }
    load()
    return () => { active = false }
  }, [supabase])

  function buildSuggestedQuery(data: BeautyPrefs | null | undefined) {
    if (!data) return ''
    const hair = data.hair_type?.replace('_', ' ')
    const concern = data.concerns?.[0]
    if (hair && concern) return `best ${hair} shampoo for ${concern}`
    if (hair) return `best ${hair} hair products`
    if (concern) return `products for ${concern}`
    return ''
  }

  async function handleSearch(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!searchInput.trim()) {
      setError('Enter something to search.')
      return
    }
    setSearching(true)
    setError('')
    setResults([])
    try {
      const query = encodeURIComponent(searchInput.trim())
      const res = await authFetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/search?query=${query}`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResults(data.validated ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <p className="text-sm text-white/60">Signed in as</p>
        <h1 className="text-2xl font-semibold">{email}</h1>
      </div>

      <section className="rounded-2xl border border-white/10 p-4 space-y-2">
        <h2 className="text-lg font-semibold">Your preferences</h2>
        {loadingPrefs ? (
          <p className="text-sm text-white/60">Loading…</p>
        ) : prefs ? (
          <ul className="text-sm space-y-1">
            <li><strong>Hair:</strong> {prefs.hair_type?.replace('_', ' ') || 'Not set'}</li>
            <li><strong>Concerns:</strong> {prefs.concerns?.length ? prefs.concerns.join(', ') : 'Not set'}</li>
            <li><strong>Price:</strong> {prefs.price_range || 'Not set'}</li>
          </ul>
        ) : (
          <p className="text-sm text-white/60">You have not completed onboarding yet.</p>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Search Reddit-backed picks</h2>
        <form onSubmit={handleSearch} className="space-y-3">
          <input
            className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="e.g. best products for dry curly hair"
            value={searchInput}
            onChange={e => setSearchInput(e.target.value)}
          />
          {error && <p className="text-sm text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={searching}
            className="rounded-xl bg-indigo-600 text-white px-4 py-2 disabled:opacity-50"
          >
            {searching ? 'Searching…' : 'Find products'}
          </button>
        </form>
      </section>

      {!!results.length && (
        <section className="space-y-4">
          <h3 className="text-lg font-semibold">Community favorites</h3>
          <div className="space-y-3">
            {results.map((r, idx) => (
              <article key={`${r.name}-${idx}`} className="rounded-2xl border border-white/10 p-4 space-y-2 bg-white/5">
                <div className="flex items-center justify-between">
                  <h4 className="text-base font-semibold">{r.name}</h4>
                  {typeof r.score === 'number' && (
                    <span className="text-xs text-white/60">Score {r.score.toFixed(2)}</span>
                  )}
                </div>
                {r.summary && <p className="text-sm text-white/80">{r.summary}</p>}
                {r.pros?.length && (
                  <div>
                    <p className="text-xs uppercase tracking-wide text-emerald-400">Pros</p>
                    <ul className="text-sm list-disc ml-4">
                      {r.pros.map((p, i) => <li key={`pro-${i}`}>{p}</li>)}
                    </ul>
                  </div>
                )}
                {r.cons?.length && (
                  <div>
                    <p className="text-xs uppercase tracking-wide text-rose-400">Cons</p>
                    <ul className="text-sm list-disc ml-4">
                      {r.cons.map((c, i) => <li key={`con-${i}`}>{c}</li>)}
                    </ul>
                  </div>
                )}
                {r.urls?.length && (
                  <div className="flex flex-wrap gap-2 text-xs">
                    {r.urls.map((url, i) => (
                      <a key={`url-${i}`} href={url} target="_blank" rel="noreferrer" className="text-indigo-300 underline">
                        Source {i + 1}
                      </a>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
