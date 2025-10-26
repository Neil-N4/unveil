import type { ReactNode } from 'react'
import './globals.css'

export const metadata = {
  title: 'Beauty Buddy',
  description: 'Personalized hair + beauty recommendations powered by Supabase + Reddit.',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
