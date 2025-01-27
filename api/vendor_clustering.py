# cluster to be number
# 
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from rapidfuzz.distance import Levenshtein
import openai
from pydantic import BaseModel

class VendorMatch(BaseModel):
    vendor_name: str
    cluster: int
    recommendation: str
    confidence: float

class VendorClusterizer:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.patterns = {
            r"AMAZON(?:\s+(?:MKTPL|MARK|RETA|GROCE|TIPS|MAR|RET|PRIME|WEB|MARKETPLACE))?\*?.*": 1,  # Amazon Retail
            r"AMAZON(?:\s+WEB\s+SERVICES?|\.COM/AWS).*": 2,  # Amazon Web Services
            r"(?:AWS|Amazon\s+Web\s+Services?)(?:\s*,?\s*Inc\.?)?": 2  # Amazon Web Services
        }
        self.pattern_names = {
            1: "Amazon Retail",
            2: "Amazon Web Services"
        }
        self.next_cluster_id = max(self.patterns.values()) + 1
        openai.api_key = openai_api_key
        
    def _pattern_match(self, vendor_name: str) -> Optional[Tuple[int, float]]:
        """Try to match vendor name against known patterns."""
        for pattern, cluster in self.patterns.items():
            if re.match(pattern, vendor_name, re.IGNORECASE):
                return cluster, 1.0
        return None

    def _preprocess_vendor_name(self, name: str) -> str:
        """Preprocess vendor name for better matching."""
        # Remove common transaction-specific parts
        name = re.sub(r'\*[A-Z0-9-]+$', '', name)
        name = re.sub(r'\s+\d{3}-\d{6}$', '', name)
        name = re.sub(r'\s+\d{5}$', '', name)
        # Remove special characters and normalize spaces
        name = re.sub(r'[,.*]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip().upper()
        
    def _fuzzy_cluster(self, vendors: List[str], min_similarity: float = 0.6) -> Dict[int, List[str]]:
        """Cluster vendors using simple similarity-based clustering."""
        clusters: Dict[int, List[str]] = {}
        processed_vendors = [self._preprocess_vendor_name(v) for v in vendors]
        
        # Initialize with first vendor
        if vendors:
            clusters[self.next_cluster_id] = [vendors[0]]
            self.next_cluster_id += 1
            
        # For each remaining vendor
        for i, vendor in enumerate(vendors[1:], 1):
            max_similarity = 0
            best_cluster = None
            
            # Compare with each existing cluster
            for cluster_id, cluster_vendors in clusters.items():
                # Calculate average similarity with cluster members
                similarities = []
                for cluster_vendor in cluster_vendors:
                    similarity = Levenshtein.normalized_similarity(
                        processed_vendors[i],
                        self._preprocess_vendor_name(cluster_vendor)
                    )
                    # Boost similarity if names start with same word
                    if processed_vendors[i].split()[0] == self._preprocess_vendor_name(cluster_vendor).split()[0]:
                        similarity = (similarity + 1) / 2
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    best_cluster = cluster_id
            
            # Add to best cluster if similarity is high enough, otherwise create new cluster
            if max_similarity >= min_similarity and best_cluster is not None:
                clusters[best_cluster].append(vendor)
            else:
                clusters[self.next_cluster_id] = [vendor]
                self.next_cluster_id += 1
        
        return clusters
    
    async def _get_llm_cluster_name(self, vendors: List[str]) -> str:
        """Use LLM to suggest a cluster name for a group of vendors."""
        prompt = f"""Given these vendor names that appear to be related:
{', '.join(vendors)}

Suggest a single standardized vendor name that best represents all of them.
The name should be clear and consistent with common vendor naming conventions.
Respond with ONLY the suggested name, nothing else."""
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    async def process_vendors(self, vendor_names: List[str]) -> List[VendorMatch]:
        """Process a list of vendor names and return clustering results."""
        results = []
        processed_vendors = set()
        
        # First try pattern matching
        for vendor in vendor_names:
            pattern_match = self._pattern_match(vendor)
            if pattern_match:
                cluster_id, confidence = pattern_match
                results.append(VendorMatch(
                    vendor_name=vendor,
                    cluster=cluster_id,
                    recommendation=self.pattern_names[cluster_id],
                    confidence=confidence
                ))
                processed_vendors.add(vendor)
        
        # Then try fuzzy clustering on remaining vendors
        remaining_vendors = [v for v in vendor_names if v not in processed_vendors]
        if remaining_vendors:
            clusters = self._fuzzy_cluster(remaining_vendors)
            
            # Process each cluster
            for cluster_id, cluster_vendors in clusters.items():
                # Get cluster name from LLM if available
                cluster_name = await self._get_llm_cluster_name(cluster_vendors)
                
                # Calculate confidence based on average similarity within cluster
                similarities = []
                processed_vendors_in_cluster = [self._preprocess_vendor_name(v) for v in cluster_vendors]
                for i, v1 in enumerate(processed_vendors_in_cluster):
                    for v2 in processed_vendors_in_cluster[i+1:]:
                        similarity = Levenshtein.normalized_similarity(v1, v2)
                        if v1.split()[0] == v2.split()[0]:
                            similarity = (similarity + 1) / 2
                        similarities.append(similarity)
                
                confidence = np.mean(similarities) if similarities else 0.75
                
                for vendor in cluster_vendors:
                    results.append(VendorMatch(
                        vendor_name=vendor,
                        cluster=cluster_id,
                        recommendation=cluster_name,
                        confidence=confidence
                    ))
                    processed_vendors.add(vendor)
        
        return results 