import openai
import re
from rapidfuzz.distance import Levenshtein
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel

    def _preprocess_vendor_name(self, name: str) -> str:
        """Clean and standardize vendor name for better matching."""
        name = name.upper()
        name = name.replace('*', ' ')
        name = name.replace(',', ' ')
        name = ' '.join(name.split())
        return name

    def _pattern_match(self, vendor: str) -> Optional[Tuple[int, float]]:
        """Try to match vendor name against known patterns."""
        for cluster_id, pattern in self.patterns.items():
            if re.match(pattern, vendor, re.IGNORECASE):
                return cluster_id, 1.0
        return None

    def _calculate_similarity(self, v1: str, v2: str) -> float:
        """Calculate similarity between two vendor names."""
        v1_clean = self._preprocess_vendor_name(v1)
        v2_clean = self._preprocess_vendor_name(v2)
        similarity = Levenshtein.normalized_similarity(v1_clean, v2_clean)
        
        # Boost similarity if vendors start with the same word
        if v1_clean.split()[0] == v2_clean.split()[0]:
            similarity = (similarity + 1) / 2
            
        return similarity

    def _fuzzy_cluster(self, vendors: List[str], min_similarity: float = 0.6) -> Dict[int, List[str]]:
        """Cluster vendors based on name similarity."""
        clusters: Dict[int, List[str]] = {}
        processed = set()

        for i, vendor in enumerate(vendors):
            if vendor in processed:
                continue

            cluster_id = self.next_cluster_id
            cluster = [vendor]
            processed.add(vendor)

            # Find similar vendors
            for other_vendor in vendors[i + 1:]:
                if other_vendor not in processed:
                    similarity = self._calculate_similarity(vendor, other_vendor)
                    if similarity >= min_similarity:
                        cluster.append(other_vendor)
                        processed.add(other_vendor)

            if len(cluster) > 0:
                clusters[cluster_id] = cluster
                self.next_cluster_id += 1

        return clusters

    async def process_vendors(self, vendors: List[str]) -> List[VendorMatch]:
        """Process a list of vendor names and return clustering results."""
        results = []
        processed = set()

        # First try pattern matching
        for vendor in vendors:
            pattern_match = self._pattern_match(vendor)
            if pattern_match:
                cluster_id, confidence = pattern_match
                results.append(VendorMatch(
                    vendor=vendor,
                    cluster=cluster_id,
                    recommendation=self.pattern_names[cluster_id],
                    confidence=confidence
                ))
                processed.add(vendor)

        # Then try fuzzy clustering on remaining vendors
        remaining_vendors = [v for v in vendors if v not in processed]
        if remaining_vendors:
            clusters = self._fuzzy_cluster(remaining_vendors)

            # Process each cluster
            for cluster_id, cluster_vendors in clusters.items():
                # Get cluster name from LLM
                cluster_name = await self._get_cluster_name(cluster_vendors)

                # Calculate confidence based on average similarity
                similarities = []
                for i, v1 in enumerate(cluster_vendors):
                    for v2 in cluster_vendors[i + 1:]:
                        similarity = self._calculate_similarity(v1, v2)
                        similarities.append(similarity)

                confidence = sum(similarities) / len(similarities) if similarities else 0.75

                for vendor in cluster_vendors:
                    results.append(VendorMatch(
                        vendor=vendor,
                        cluster=cluster_id,
                        recommendation=cluster_name,
                        confidence=confidence
                    ))
                    processed.add(vendor)

        # Handle any remaining unmatched vendors
        for vendor in vendors:
            if vendor not in processed:
                results.append(VendorMatch(
                    vendor=vendor,
                    cluster=self.next_cluster_id,
                    recommendation=vendor,
                    confidence=0.5
                ))
                self.next_cluster_id += 1

        return results 