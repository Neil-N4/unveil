/*
  # Create beauty preferences table

  1. New Tables
    - `beauty_preferences`
      - `id` (uuid, primary key) - Unique identifier for each preference record
      - `user_id` (uuid, foreign key to auth.users) - Links to the authenticated user
      - `hair_type` (text, nullable) - User's hair type (e.g., oily, dry, curly, etc.)
      - `concerns` (text[], nullable) - Array of hair concerns (e.g., dandruff, frizz, etc.)
      - `price_range` (text, nullable) - Preferred price range (drugstore, mid, premium)
      - `completed` (boolean, default false) - Whether onboarding is completed
      - `created_at` (timestamptz) - When the record was created
      - `updated_at` (timestamptz) - When the record was last updated

  2. Security
    - Enable RLS on `beauty_preferences` table
    - Add policy for users to read their own preferences
    - Add policy for users to insert their own preferences
    - Add policy for users to update their own preferences
    - Add policy for users to delete their own preferences

  3. Notes
    - Each user can have only one preferences record (enforced by unique constraint)
    - The user_id references auth.users with cascade on delete
    - Timestamps are automatically managed via default values and trigger
*/

-- Create the beauty_preferences table
CREATE TABLE IF NOT EXISTS beauty_preferences (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  hair_type text,
  concerns text[] DEFAULT '{}',
  price_range text,
  completed boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id)
);

-- Enable RLS
ALTER TABLE beauty_preferences ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users to manage their own preferences
CREATE POLICY "Users can read own preferences"
  ON beauty_preferences
  FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own preferences"
  ON beauty_preferences
  FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own preferences"
  ON beauty_preferences
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own preferences"
  ON beauty_preferences
  FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_beauty_preferences_updated_at
  BEFORE UPDATE ON beauty_preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Create index for faster lookups by user_id
CREATE INDEX IF NOT EXISTS idx_beauty_preferences_user_id ON beauty_preferences(user_id);