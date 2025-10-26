-- Run this in Supabase SQL Editor

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  created_at timestamptz default now(),
  display_name text,
  avatar_url text
);

create table if not exists public.beauty_preferences (
  user_id uuid primary key references auth.users(id) on delete cascade,
  hair_type text check (hair_type in ('oily','dry','normal','curly','wavy','thick','fine','color_treated')),
  concerns text[],
  price_range text,
  brand_avoid text[],
  completed boolean default false,
  updated_at timestamptz default now()
);

alter table public.profiles enable row level security;
alter table public.beauty_preferences enable row level security;

drop policy if exists "profiles self" on public.profiles;
create policy "profiles self" on public.profiles for all
using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "prefs self" on public.beauty_preferences;
create policy "prefs self" on public.beauty_preferences for all
using (auth.uid() = user_id) with check (auth.uid() = user_id);

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (user_id, email) values (new.id, new.email);
  return new;
end; $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();
