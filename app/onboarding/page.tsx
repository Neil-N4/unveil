'use client'
import { useEffect, useMemo, useState } from 'react'
import { createClient } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

const HAIR_TYPES = ['oily','dry','normal','curly','wavy','thick','fine','color_treated']
const CONCERNS = ['dandruff','build-up','frizz','flat','hard water','damage']

export default function Onboarding() {
  const supabase = useMemo(() => createClient(), [])
  const router = useRouter()
  const [hair, setHair] = useState('')
  const [concerns, setConcerns] = useState<string[]>([])
  const [price, setPrice] = useState('drugstore')
  const [status, setStatus] = useState<'idle'|'saving'|'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    let active = true
    async function loadExisting() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        router.replace('/sign-in')
        return
      }
      const { data } = await supabase
        .from('beauty_preferences')
        .select('hair_type, concerns, price_range')
        .eq('user_id', user.id)
        .maybeSingle()

      if (!active || !data) return
      setHair(data.hair_type ?? '')
      setConcerns(data.concerns ?? [])
      setPrice(data.price_range ?? 'drugstore')
    }
    loadExisting()
    return () => { active = false }
  }, [router, supabase])

  async function save() {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return router.push('/sign-in')
    setStatus('saving')
    setErrorMessage('')
    const { error } = await supabase.from('beauty_preferences').upsert({
      user_id: user.id, hair_type: hair || null, concerns, price_range: price, completed: true
    })
    if (error) {
      setStatus('error')
      setErrorMessage(error.message)
      return
    }
    setStatus('idle')
    router.push('/discover')
  }

  return (
    <div className="mx-auto max-w-2xl p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Tell us about your hair</h1>

      <div className="grid grid-cols-2 gap-3">
        {HAIR_TYPES.map(v=>(
          <button key={v} onClick={()=>setHair(v)}
            className={`rounded-xl border px-3 py-2 ${hair===v?'border-indigo-400 bg-indigo-500/10':'border-white/10'}`}>
            {v.replace('_',' ')}
          </button>
        ))}
      </div>

      <div>
        <p className="mb-2">Concerns</p>
        {CONCERNS.map(c=>(
          <label key={c} className="mr-3 text-sm">
            <input type="checkbox" checked={concerns.includes(c)}
              onChange={()=>setConcerns(prev=>prev.includes(c)?prev.filter(x=>x!==c):[...prev, c])}/> {c}
          </label>
        ))}
      </div>

      <div>
        <p className="mb-2">Price range</p>
        {['drugstore','mid','premium'].map(p=>(
          <label key={p} className="mr-3 text-sm">
            <input type="radio" name="price" checked={price===p} onChange={()=>setPrice(p)} /> {p}
          </label>
        ))}
      </div>

      {errorMessage && <p className="text-red-500 text-sm">{errorMessage}</p>}
      <button
        onClick={save}
        disabled={status === 'saving'}
        className="rounded-xl bg-indigo-600 text-white px-4 py-2 disabled:opacity-50"
      >
        {status === 'saving' ? 'Saving…' : 'Save & continue'}
      </button>
    </div>
  )
}
