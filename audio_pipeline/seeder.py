# audio_pipeline/seeder.py

import os
import tempfile
from supabase import create_client, Client

# Import our custom modules
from core.separator import StemSeparator
from core.beat_slicer import extract_rhythmic_loops
from core.aifm_engine import initialize_aifm, extract_audio_embedding
from core.classifier import ZeroShotAudioAnalyzer

# Initialize Supabase client
# Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are in your .env
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Required for bypassing RLS during backend seeding
)

def seed_audio_track(raw_audio_path: str, track_id: str):
    """
    End-to-end ingestion: separates vocals, slices loops, extracts vectors, 
    calculates coordinates, and seeds the Supabase database.
    """
    print(f"\n--- Starting Ingestion for {track_id} ---")
    
    # 1. Initialize ML Models (Load to GPU once)
    initialize_aifm()
    separator = StemSeparator()
    analyzer = ZeroShotAudioAnalyzer()

    # 2. Allocate the self-destructing temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        mix_dir = os.path.join(temp_dir, "mix")
        chunks_dir = os.path.join(temp_dir, "chunks")
        os.makedirs(mix_dir)
        os.makedirs(chunks_dir)
        
        # 3. DSP Phase: Separate and Slice
        print(f"[{track_id}] Extracting instrumental mix...")
        instrumental_path = separator.create_instrumental(raw_audio_path, output_dir=mix_dir)
        
        print(f"[{track_id}] Slicing instrumental into rhythmic loops...")
        slices_metadata = extract_rhythmic_loops(instrumental_path, track_id, output_dir=chunks_dir)
        
        print(f"[{track_id}] Generated {len(slices_metadata)} slices. Starting vectorization...")

        # 4. ML & Database Phase: Embed, Classify, and Upload
        for idx, chunk in enumerate(slices_metadata):
            filepath = chunk["filepath"]
            filename = os.path.basename(filepath)
            storage_path = f"audio_assets/{filename}"
            
            # A. Extract 1024D Vector
            vector = extract_audio_embedding(filepath)
            
            # B. Extract Category and [Mood, Energy] Coordinates
            analysis = analyzer.analyze(vector)
            
            # C. Upload .wav to Supabase Storage Bucket
            with open(filepath, "rb") as f:
                supabase.storage.from_("audio").upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "audio/wav"}
                )
            
            # D. Insert row into PostgreSQL pgvector table
            supabase.table("audio_library").insert({
                "track_id": track_id,
                "storage_path": storage_path,
                "duration": chunk["duration"],
                "bpm": chunk["bpm"],
                "category": analysis["category"],
                "ui_coordinate": analysis["ui_coordinate"],
                "embedding": vector
            }).execute()
            
            print(f"  -> Uploaded {filename} | {analysis['category']} | Coords: {analysis['ui_coordinate']}")

    print(f"--- Ingestion Complete for {track_id}. Workspace erased. ---\n")

if __name__ == "__main__":
    # Example usage:
    # Requires an empty "audio" storage bucket created in your Supabase dashboard
    seed_audio_track("data/raw/cyberpunk_chase_01.mp3", "cyberpunk_01")