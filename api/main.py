from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from api.vendor_clustering import VendorClusterizer, VendorMatch
import os

app = FastAPI(title="Vendor Clustering Service")

class VendorRequest(BaseModel):
    vendor_names: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "vendor_names": [
                    "AMAZON MKTPL*XM1EJ9M33",
                    "AMAZON.COM",
                    "AMAZON WEB SERVICES"
                ]
            }
        }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/process-vendors", response_model=List[VendorMatch])
async def process_vendors(request: VendorRequest):
    """
    Process a list of vendor names and return clustering results.
    
    Each result includes:
    - vendor_name: Original vendor name
    - cluster: Assigned cluster name
    - recommendation: Recommended standardized name
    - confidence: Confidence score (0-1) for the clustering
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is not set")
            
        clusterizer = VendorClusterizer(openai_api_key=api_key)
        results = await clusterizer.process_vendors(request.vendor_names)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) 