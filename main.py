from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torchaudio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cdist
import shutil
import os

app = FastAPI()

embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def extract_embedding(audio, sample_rate):
    return embedding_model(audio[None])

def compare_embeddings(embedding1, embedding2):
    similarity = 1 - cdist(embedding1, embedding2, metric="cosine")
    return similarity[0][0]

@app.post("/verify-voice/")
async def verify_voice(suspect: UploadFile = File(...), test: UploadFile = File(...)):
    suspect_path = f"temp_{suspect.filename}"
    test_path = f"temp_{test.filename}"

    with open(suspect_path, "wb") as f:
        shutil.copyfileobj(suspect.file, f)

    with open(test_path, "wb") as f:
        shutil.copyfileobj(test.file, f)

    try:
        suspect_audio, sr1 = load_audio(suspect_path)
        test_audio, sr2 = load_audio(test_path)

        if sr1 != sr2:
            return JSONResponse(status_code=400, content={"message": "Sample rates do not match"})

        emb1 = extract_embedding(suspect_audio, sr1)
        emb2 = extract_embedding(test_audio, sr2)
        similarity = compare_embeddings(emb1, emb2)

        # Clean up temp files
        os.remove(suspect_path)
        os.remove(test_path)

        threshold = 0.8
        if similarity >= threshold:
            return {"message": "Voice matched", "confidence": round(similarity * 100, 2)}
        else:
            return {"message": "Voice not matched", "confidence": round(similarity * 100, 2)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
