// src/features/video/api/fetch-trajectory.ts

import { supabaseClient } from "@/lib/supabase";

export async function fetchVideoTrajectory(videoId: string) {
  const { data, error } = await supabaseClient
    .from("videos")
    .select("trajectory")
    .eq("id", videoId)
    .single(); // Forces Supabase to return a direct object instead of an array

  if (error) {
    throw new Error(`Database Fetch Error: ${error.message}`);
  }
  
  return data?.trajectory;
}