from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from supabase import create_client, Client

app = FastAPI()

# 1. Allow the frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load your newly trained 89% accurate model!
model = joblib.load('mood_model.pkl')

# 3. Connect to Supabase
url: str = "https://ksomngfnsoumzmrcofzk.supabase.co/rest/v1/"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imtzb21uZ2Zuc291bXptcmNvZnprIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA2MDc5NjcsImV4cCI6MjA5NjE4Mzk2N30.halbTgmq_lQMz_2jcNBCcrpyfzpXhrMinYw9yOBMDDk"
supabase: Client = create_client(url, key)

# 4. Define the exact structure of the data coming from the frontend
class KeystrokeData(BaseModel):
    device_type: str
    mean_dwell_ms: float
    std_dwell_ms: float
    mean_flight_ms: float
    std_flight_ms: float
    typing_speed_wpm: float
    error_rate: float
    rhythm_variance: float
    pause_count: int            # Upgraded feature
    hold_intensity: float       # Upgraded feature
    digraph_latency_ms: float   # Upgraded feature
    trigraph_latency_ms: float  # Upgraded feature
    user_reported_mood: str = None 

@app.post("/process_keystrokes")
def process_keystrokes(data: KeystrokeData):
    if data.device_type == 'mobile':
        # MOBILE: Do not predict. Save the raw data + reported mood to Supabase!
        mobile_data = data.dict()
        
        # Insert into the Supabase table
        supabase.table("mobile_keystrokes").insert(mobile_data).execute()
        
        return {"status": "success", "message": "Mobile data collected and sent to Supabase."}
    
    else:
        # DESKTOP: Make a prediction using the model
        input_features = [[
            data.mean_dwell_ms, data.std_dwell_ms, data.mean_flight_ms, 
            data.std_flight_ms, data.typing_speed_wpm, data.error_rate, 
            data.rhythm_variance, data.pause_count, data.hold_intensity, 
            data.digraph_latency_ms, data.trigraph_latency_ms
        ]]
        
        prediction = model.predict(input_features)[0]
        return {"status": "success", "predicted_mood": prediction}