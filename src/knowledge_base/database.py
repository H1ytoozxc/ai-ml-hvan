"""
Knowledge Base
База знаний для хранения всех оцененных архитектур
"""

import sqlite3
import json
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from ..search_space.search_space import ArchitectureGenome


class KnowledgeBase:
    """Persistent storage for evaluated architectures"""
    
    def __init__(self, db_path: str = "./evo_runs/knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database schema"""
        
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Architectures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS architectures (
                genome_id TEXT PRIMARY KEY,
                genome_json TEXT NOT NULL,
                generation INTEGER,
                created_at TIMESTAMP,
                parent_ids TEXT
            )
        """)
        
        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                genome_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                timestamp TIMESTAMP,
                FOREIGN KEY (genome_id) REFERENCES architectures(genome_id)
            )
        """)
        
        # Novelty scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS novelty_scores (
                genome_id TEXT PRIMARY KEY,
                architectural_novelty REAL,
                behavioral_novelty REAL,
                combined_novelty REAL,
                activation_profile BLOB,
                FOREIGN KEY (genome_id) REFERENCES architectures(genome_id)
            )
        """)
        
        # Pareto fronts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pareto_fronts (
                front_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                stage TEXT,
                genome_ids TEXT,
                objectives_json TEXT,
                timestamp TIMESTAMP
            )
        """)
        
        # Meta-learning statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta_stats (
                generation INTEGER,
                mutation_type TEXT,
                success_rate REAL,
                avg_improvement REAL,
                timestamp TIMESTAMP
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_genome_generation ON architectures(generation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_genome ON evaluations(genome_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_eval_stage ON evaluations(stage)")
        
        self.conn.commit()
    
    def add_architecture(self, genome: ArchitectureGenome) -> bool:
        """Add architecture to database"""
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO architectures 
                (genome_id, genome_json, generation, created_at, parent_ids)
                VALUES (?, ?, ?, ?, ?)
            """, (
                genome.genome_id,
                genome.to_json(),
                genome.generation,
                datetime.now(),
                json.dumps(genome.parent_ids)
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error adding architecture: {e}")
            return False
    
    def add_evaluation(self, 
                      genome_id: str,
                      stage: str,
                      metrics: Dict[str, float]) -> bool:
        """Add evaluation results"""
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO evaluations 
                (genome_id, stage, metrics_json, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                genome_id,
                stage,
                json.dumps(metrics),
                datetime.now()
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error adding evaluation: {e}")
            return False
    
    def add_novelty_score(self,
                         genome_id: str,
                         novelty_metrics: Dict[str, float],
                         activation_profile: Optional[np.ndarray] = None) -> bool:
        """Add novelty scores"""
        
        try:
            cursor = self.conn.cursor()
            
            # Serialize activation profile
            profile_blob = pickle.dumps(activation_profile) if activation_profile is not None else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO novelty_scores
                (genome_id, architectural_novelty, behavioral_novelty, 
                 combined_novelty, activation_profile)
                VALUES (?, ?, ?, ?, ?)
            """, (
                genome_id,
                novelty_metrics.get("architectural_novelty", 0),
                novelty_metrics.get("behavioral_novelty", 0),
                novelty_metrics.get("combined_novelty", 0),
                profile_blob
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error adding novelty score: {e}")
            return False
    
    def save_pareto_front(self,
                         generation: int,
                         stage: str,
                         genome_ids: List[str],
                         objectives: List[Dict[str, float]]) -> bool:
        """Save Pareto front"""
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO pareto_fronts
                (generation, stage, genome_ids, objectives_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                generation,
                stage,
                json.dumps(genome_ids),
                json.dumps(objectives),
                datetime.now()
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error saving Pareto front: {e}")
            return False
    
    def add_meta_stat(self,
                     generation: int,
                     mutation_type: str,
                     success_rate: float,
                     avg_improvement: float) -> bool:
        """Add meta-learning statistics"""
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO meta_stats
                (generation, mutation_type, success_rate, avg_improvement, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                generation,
                mutation_type,
                success_rate,
                avg_improvement,
                datetime.now()
            ))
            
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"Error adding meta stat: {e}")
            return False
    
    def get_architecture(self, genome_id: str) -> Optional[ArchitectureGenome]:
        """Retrieve architecture by ID"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT genome_json FROM architectures WHERE genome_id = ?
        """, (genome_id,))
        
        result = cursor.fetchone()
        if result:
            genome_dict = json.loads(result[0])
            return ArchitectureGenome.from_dict(genome_dict)
        
        return None
    
    def get_evaluations(self, genome_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation results for architecture"""
        
        cursor = self.conn.cursor()
        
        if stage:
            cursor.execute("""
                SELECT metrics_json, timestamp FROM evaluations 
                WHERE genome_id = ? AND stage = ?
                ORDER BY timestamp DESC
            """, (genome_id, stage))
        else:
            cursor.execute("""
                SELECT stage, metrics_json, timestamp FROM evaluations 
                WHERE genome_id = ?
                ORDER BY timestamp DESC
            """, (genome_id,))
        
        results = []
        for row in cursor.fetchall():
            if stage:
                metrics_json, timestamp = row
                results.append({
                    "metrics": json.loads(metrics_json),
                    "timestamp": timestamp
                })
            else:
                stage_name, metrics_json, timestamp = row
                results.append({
                    "stage": stage_name,
                    "metrics": json.loads(metrics_json),
                    "timestamp": timestamp
                })
        
        return results
    
    def get_top_architectures(self, 
                             stage: str,
                             metric: str = "accuracy",
                             limit: int = 10) -> List[ArchitectureGenome]:
        """Get top performing architectures"""
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT a.genome_json, e.metrics_json
            FROM architectures a
            JOIN evaluations e ON a.genome_id = e.genome_id
            WHERE e.stage = ?
            ORDER BY json_extract(e.metrics_json, '$.' || ?) DESC
            LIMIT ?
        """, (stage, metric, limit))
        
        genomes = []
        for row in cursor.fetchall():
            genome_dict = json.loads(row[0])
            genome = ArchitectureGenome.from_dict(genome_dict)
            genomes.append(genome)
        
        return genomes
    
    def get_population_by_generation(self, generation: int) -> List[ArchitectureGenome]:
        """Get all architectures from a generation"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT genome_json FROM architectures WHERE generation = ?
        """, (generation,))
        
        genomes = []
        for row in cursor.fetchall():
            genome_dict = json.loads(row[0])
            genome = ArchitectureGenome.from_dict(genome_dict)
            genomes.append(genome)
        
        return genomes
    
    def get_activation_profiles(self, genome_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get activation profiles for genomes"""
        
        cursor = self.conn.cursor()
        
        profiles = {}
        for genome_id in genome_ids:
            cursor.execute("""
                SELECT activation_profile FROM novelty_scores WHERE genome_id = ?
            """, (genome_id,))
            
            result = cursor.fetchone()
            if result and result[0]:
                profiles[genome_id] = pickle.loads(result[0])
        
        return profiles
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total architectures
        cursor.execute("SELECT COUNT(*) FROM architectures")
        stats["total_architectures"] = cursor.fetchone()[0]
        
        # Total evaluations
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        stats["total_evaluations"] = cursor.fetchone()[0]
        
        # Evaluations by stage
        cursor.execute("""
            SELECT stage, COUNT(*) FROM evaluations GROUP BY stage
        """)
        stats["evaluations_by_stage"] = dict(cursor.fetchall())
        
        # Current generation
        cursor.execute("SELECT MAX(generation) FROM architectures")
        max_gen = cursor.fetchone()[0]
        stats["max_generation"] = max_gen if max_gen is not None else 0
        
        return stats
    
    def export_top_models(self, output_dir: str, top_n: int = 10):
        """Export top models to JSON files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get top architectures from final stage
        top_genomes = self.get_top_architectures(
            stage="Stage3_Full_Validation",
            metric="accuracy",
            limit=top_n
        )
        
        for i, genome in enumerate(top_genomes):
            output_path = os.path.join(output_dir, f"top_model_{i+1}.json")
            with open(output_path, 'w') as f:
                f.write(genome.to_json())
        
        print(f"Exported {len(top_genomes)} top models to {output_dir}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PerformanceCache:
    """In-memory cache for quick lookups"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def add(self, genome_id: str, stage: str, metrics: Dict[str, float]):
        """Add to cache"""
        
        key = f"{genome_id}_{stage}"
        self.cache[key] = metrics
        
        # Simple LRU: remove oldest if over size
        if len(self.cache) > self.max_size:
            # Remove first item (oldest)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
    
    def get(self, genome_id: str, stage: str) -> Optional[Dict[str, float]]:
        """Get from cache"""
        
        key = f"{genome_id}_{stage}"
        return self.cache.get(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
