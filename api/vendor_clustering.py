import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from rapidfuzz.distance import Levenshtein
import openai
from pydantic import BaseModel
import asyncio

class VendorMatch(BaseModel):
    vendor_name: str
    cluster: int
    recommendation: str
    confidence: float

class VendorClusterizer:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.next_cluster_id = 1
        
        # Pattern to standardized name mapping
        self.pattern_mapping = {
            # Amazon patterns
            "AMAZON WEB SERVICES": "Amazon Web Services",
            "AMAZON.COM": "Amazon.com",
            "AMAZON GROCE": "Amazon Retail",
            "AMAZON GROCERY": "Amazon Retail",
            "AMAZON MAR": "Amazon Retail",
            "AMAZON MARK": "Amazon Retail",
            "AMAZON MARKETPLACE": "Amazon Retail",
            "AMAZON MKTPL": "Amazon Retail",
            "AMAZON PRIME": "Amazon Retail",
            "AMAZON RET": "Amazon Retail",
            "AMAZON RETA": "Amazon Retail",
            "AMAZON TIPS": "Amazon Retail",
            "AMZ": "Amazon Retail",
            "AMZN MKTP": "Amazon Retail",
            "WWW.AMAZON": "Amazon Retail",
            
            # # Apple patterns
            # "APPLE.COM": "Apple Retail",
            # "APPLE STORE": "Apple Retail",
            # "APPLE ADS": "Apple Ads",
            # "APPLE": "Apple",
            
            # # Canva patterns
            # "CANVA": "Canva",
            
            # # Wendy's patterns
            # "WENDYS": "Wendy's",
            
            # # Zendesk patterns
            # "ZENDESK INC": "Zendesk, Inc.",
            # "ZENDESK": "Zendesk, Inc."
        }
        
    def _extract_base_pattern(self, vendor_name: str) -> Optional[Tuple[str, float]]:
        """Extract the base pattern and confidence from a vendor name."""
        # Normalize the name first
        name = self._preprocess_vendor_name(vendor_name)
        
        # Try to match against known patterns
        for base_pattern in self.pattern_mapping.keys():
            # Check for exact match
            if name == base_pattern:
                return base_pattern, 1.0
            
            # Check if name starts with pattern (e.g., "AMAZON MKTPL*123456")
            if name.startswith(base_pattern + " ") or name.startswith(base_pattern + "*"):
                # Calculate confidence based on how much of the original string matches
                pattern_words = set(base_pattern.split())
                name_words = set(name.split())
                common_words = pattern_words.intersection(name_words)
                confidence = len(common_words) / max(len(pattern_words), len(name_words))
                # Boost confidence for pattern matches
                confidence = min((confidence + 0.7) / 2 + 0.5, 1.0)
                return base_pattern, confidence
        
        # For unmatched patterns, return the first word with low confidence
        words = name.split()
        if words:
            return words[0], 0.3
        return None

    async def _get_llm_cluster_names(self, vendor_groups: List[List[str]]) -> List[str]:
        """Use LLM to suggest cluster names for multiple groups of vendors in parallel."""
        async def get_single_name(vendors: List[str]) -> str:
            prompt = f"""Given these vendor names that appear to be related:
{', '.join(vendors)}

Suggest a single standardized vendor name that best represents all of them.
The name should be clear and consistent with common vendor naming conventions.
Respond with ONLY the suggested name, nothing else."""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        
        # Create tasks for all vendor groups
        tasks = [get_single_name(vendors) for vendors in vendor_groups]
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks)
        return results

    async def _pattern_match(self, vendor_names: List[str]) -> List[Optional[Tuple[float, str]]]:
        """Try to match multiple vendor names against known patterns in parallel."""
        results = [None] * len(vendor_names)
        llm_vendors_by_prefix = {}  # Group vendors by prefix for LLM processing
        
        # First pass: handle known patterns and group others by prefix
        for i, vendor_name in enumerate(vendor_names):
            match = self._extract_base_pattern(vendor_name)
            if not match:
                continue
                
            base_pattern, confidence = match
            
            # If it's a known pattern, use the mapped recommendation
            if base_pattern in self.pattern_mapping:
                results[i] = (confidence, self.pattern_mapping[base_pattern])
            else:
                # Group by prefix for LLM processing
                processed_name = self._preprocess_vendor_name(vendor_name)
                prefix = processed_name.split()[0] if processed_name else ""
                if prefix:
                    if prefix not in llm_vendors_by_prefix:
                        llm_vendors_by_prefix[prefix] = []
                    llm_vendors_by_prefix[prefix].append((i, vendor_name))
        
        # Process all LLM requests in parallel if there are any
        if llm_vendors_by_prefix:
            # Create groups for LLM processing
            prefix_groups = []
            index_maps = []  # Keep track of which indices belong to which group
            
            for prefix, vendors in llm_vendors_by_prefix.items():
                prefix_groups.append([v[1] for v in vendors])  # Group of vendor names
                index_maps.append([v[0] for v in vendors])    # Corresponding indices
            
            # Get recommendations for all groups in parallel
            group_recommendations = await self._get_llm_cluster_names(prefix_groups)
            
            # Process results
            for indices, recommendation in zip(index_maps, group_recommendations):
                # Calculate base confidence based on group size
                base_confidence = 0.7  # Higher base confidence for prefix groups
                
                for idx in indices:
                    vendor = vendor_names[idx]
                    processed_vendor = self._preprocess_vendor_name(vendor)
                    processed_recommendation = self._preprocess_vendor_name(recommendation)
                    
                    # Calculate similarity
                    similarity = Levenshtein.normalized_similarity(processed_vendor, processed_recommendation)
                    
                    # Calculate confidence
                    if processed_vendor == processed_recommendation:
                        confidence = 1.0
                    elif processed_vendor.split()[0] == processed_recommendation.split()[0]:
                        confidence = max((similarity + 1) / 2, 0.8)
                    elif (processed_vendor.startswith(processed_recommendation) or 
                          processed_recommendation.startswith(processed_vendor)):
                        confidence = max((similarity + base_confidence) / 2, 0.7)
                    else:
                        confidence = max(similarity, base_confidence)
                    
                    results[idx] = (confidence, recommendation)
        
        return results

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
        
    def _fuzzy_cluster(self, vendors: List[str], min_similarity: float = 0.5) -> Dict[int, List[str]]:
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
                    processed_cluster_vendor = self._preprocess_vendor_name(cluster_vendor)
                    similarity = Levenshtein.normalized_similarity(
                        processed_vendors[i],
                        processed_cluster_vendor
                    )
                    # Boost similarity if names start with same word
                    if processed_vendors[i].split()[0] == processed_cluster_vendor.split()[0]:
                        similarity = (similarity + 1) / 2
                    # Additional boost for partial prefix matches
                    elif (processed_vendors[i].startswith(processed_cluster_vendor) or 
                          processed_cluster_vendor.startswith(processed_vendors[i])):
                        similarity = (similarity + 0.3)
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
    
    async def process_vendors(self, vendor_names: List[str]) -> List[VendorMatch]:
        """Process a list of vendor names and return clustering results."""
        results = []
        processed_vendors = set()
        recommendation_to_cluster = {}  # Maps recommendations to cluster IDs
        
        # First try pattern matching all vendors in parallel
        pattern_matches = await self._pattern_match(vendor_names)
        
        # Process pattern matches
        for vendor, match in zip(vendor_names, pattern_matches):
            if match:
                confidence, recommendation = match
                
                # Assign cluster ID if we haven't seen this recommendation before
                if recommendation not in recommendation_to_cluster:
                    recommendation_to_cluster[recommendation] = self.next_cluster_id
                    self.next_cluster_id += 1
                
                cluster_id = recommendation_to_cluster[recommendation]
                results.append(VendorMatch(
                    vendor_name=vendor,
                    cluster=cluster_id,
                    recommendation=recommendation,
                    confidence=confidence
                ))
                processed_vendors.add(vendor)
        
        # Then try fuzzy clustering on remaining vendors
        remaining_vendors = [v for v in vendor_names if v not in processed_vendors]
        if remaining_vendors:
            # First, group by common prefixes
            prefix_groups = {}
            for vendor in remaining_vendors:
                processed_name = self._preprocess_vendor_name(vendor)
                base_word = processed_name.split()[0] if processed_name else ""
                if base_word:
                    if base_word not in prefix_groups:
                        prefix_groups[base_word] = []
                    prefix_groups[base_word].append(vendor)
            
            # Prepare all groups for parallel LLM processing
            group_vendors = []
            group_info = []  # Store cluster_id and vendors for each group
            
            # Process each prefix group
            for prefix, vendors_in_group in prefix_groups.items():
                # Always keep vendors with the same prefix in one cluster
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                group_vendors.append(vendors_in_group)
                group_info.append((cluster_id, vendors_in_group, prefix))
            
            # Get all cluster names in parallel
            cluster_names = await self._get_llm_cluster_names(group_vendors)
            
            # Process results
            for (cluster_id, cluster_vendors, prefix), cluster_name in zip(group_info, cluster_names):
                # Calculate individual confidence for each vendor
                for vendor in cluster_vendors:
                    processed_vendor = self._preprocess_vendor_name(vendor)
                    processed_recommendation = self._preprocess_vendor_name(cluster_name)
                    
                    # Start with a higher base confidence for same-prefix groups
                    base_confidence = 0.7
                    
                    # Calculate similarity
                    similarity = Levenshtein.normalized_similarity(processed_vendor, processed_recommendation)
                    
                    # Apply boosts
                    if processed_vendor == processed_recommendation:
                        # Exact match after preprocessing
                        confidence = 1.0
                    elif processed_vendor.split()[0] == processed_recommendation.split()[0]:
                        # Same first word
                        confidence = max((similarity + 1) / 2, 0.8)
                    elif (processed_vendor.startswith(processed_recommendation) or 
                          processed_recommendation.startswith(processed_vendor)):
                        # One is prefix of the other
                        confidence = max((similarity + base_confidence) / 2, 0.7)
                    else:
                        # Use similarity but ensure it's at least the base confidence
                        confidence = max(similarity, base_confidence)
                    
                    # Ensure confidence is between 0 and 1
                    confidence = min(max(confidence, base_confidence), 1.0)
                    
                    results.append(VendorMatch(
                        vendor_name=vendor,
                        cluster=cluster_id,
                        recommendation=cluster_name,
                        confidence=confidence
                    ))
                    processed_vendors.add(vendor)
        
        return results 