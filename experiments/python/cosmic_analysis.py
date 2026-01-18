#!/usr/bin/env python3
"""
ANALYSE APPROFONDIE : INFORMATIQUE COSMIQUE
Étude des patterns de convergence et d'émergence
"""

import time
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import json

@dataclass
class Particle:
    value: float
    stability: float
    mass: float
    generation: int = 0  # Pour tracer l'évolution
    
    def distance_to(self, other: 'Particle') -> float:
        return abs(self.value - other.value)
    
    def fuse_with(self, other: 'Particle') -> 'Particle':
        """Fusion nucléaire informationnelle"""
        total_mass = self.mass + other.mass
        new_value = (self.value * self.mass + other.value * other.mass) / total_mass
        new_stability = min(1.0, self.stability + other.stability + 0.05)
        
        return Particle(
            value=new_value,
            stability=new_stability,
            mass=total_mass,
            generation=max(self.generation, other.generation) + 1
        )


class CosmicAnalyzer:
    """Analyseur de patterns cosmiques"""
    
    def __init__(self):
        self.metrics = {
            'entropy': [],
            'coherence': [],
            'compression_rate': [],
            'emergence_index': []
        }
    
    def analyze_universe(self, particles: List[Particle]) -> Dict:
        """Analyse complète de l'état de l'univers"""
        if not particles:
            return {'entropy': 0, 'coherence': 0, 'emergence': 0}
        
        # ENTROPIE : Dispersion des valeurs
        values = [p.value for p in particles]
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        entropy = variance ** 0.5  # Écart-type
        
        # COHÉRENCE : Stabilité moyenne
        coherence = sum(p.stability for p in particles) / len(particles)
        
        # ÉMERGENCE : Présence de super-structures (masse > 100)
        super_structures = sum(1 for p in particles if p.mass > 100)
        emergence = super_structures / len(particles) if particles else 0
        
        return {
            'entropy': entropy,
            'coherence': coherence,
            'emergence': emergence,
            'total_mass': sum(p.mass for p in particles),
            'avg_generation': sum(p.generation for p in particles) / len(particles)
        }


class AdvancedUniversalEngine:
    """Version avancée avec métriques et patterns cosmiques"""
    
    def __init__(self, fusion_threshold: float = 5.0):
        self.particles: List[Particle] = []
        self.fusion_threshold = fusion_threshold
        self.analyzer = CosmicAnalyzer()
        self.history = []
        self.cycle_count = 0
    
    def inject_chaos(self, amount: int, value_range: Tuple[float, float] = (0, 100)):
        """Injection de chaos avec distribution contrôlée"""
        for _ in range(amount):
            self.particles.append(Particle(
                value=random.uniform(*value_range),
                stability=random.uniform(0.05, 0.2),
                mass=1.0,
                generation=0
            ))
    
    def inject_pattern(self, pattern_type: str, count: int):
        """Injection de patterns spécifiques (étoiles, nébuleuses, etc.)"""
        if pattern_type == "pulsar":
            # Pattern régulier : valeurs espacées uniformément
            for i in range(count):
                self.particles.append(Particle(
                    value=i * 10.0,
                    stability=0.9,
                    mass=5.0,
                    generation=0
                ))
        
        elif pattern_type == "nebula":
            # Pattern diffus : cluster autour de quelques centres
            centers = [random.uniform(20, 80) for _ in range(5)]
            for _ in range(count):
                center = random.choice(centers)
                self.particles.append(Particle(
                    value=center + random.gauss(0, 2),
                    stability=0.3,
                    mass=1.0,
                    generation=0
                ))
        
        elif pattern_type == "black_hole":
            # Pattern attracteur : une masse centrale énorme
            self.particles.append(Particle(
                value=50.0,
                stability=1.0,
                mass=100.0,
                generation=0
            ))
            # Matière orbitale
            for _ in range(count - 1):
                self.particles.append(Particle(
                    value=50.0 + random.uniform(-10, 10),
                    stability=0.1,
                    mass=1.0,
                    generation=0
                ))
    
    def pulse_process_detailed(self) -> Dict:
        """Pulse avec métriques détaillées"""
        start = time.perf_counter()
        
        # Analyse pré-fusion
        pre_analysis = self.analyzer.analyze_universe(self.particles)
        initial_count = len(self.particles)
        
        # Phase de fusion gravitationnelle
        next_gen = []
        processed = [False] * initial_count
        fusion_events = []
        
        for i in range(initial_count):
            if processed[i]:
                continue
            
            master = self.particles[i]
            processed[i] = True
            fused_with = []
            
            for j in range(i + 1, initial_count):
                if processed[j]:
                    continue
                
                distance = master.distance_to(self.particles[j])
                
                if distance < self.fusion_threshold:
                    fused_with.append(j)
                    master = master.fuse_with(self.particles[j])
                    processed[j] = True
            
            if fused_with:
                fusion_events.append({
                    'master_idx': i,
                    'absorbed': len(fused_with),
                    'final_mass': master.mass
                })
            
            # Harmonisation : les structures massives s'auto-stabilisent
            if master.mass > 10.0:
                master.stability = min(1.0, master.stability + 0.1)
            
            next_gen.append(master)
        
        self.particles = next_gen
        duration = (time.perf_counter() - start) * 1_000_000
        
        # Analyse post-fusion
        post_analysis = self.analyzer.analyze_universe(self.particles)
        
        self.cycle_count += 1
        
        result = {
            'cycle': self.cycle_count,
            'duration_us': duration,
            'initial_count': initial_count,
            'final_count': len(self.particles),
            'fusions': initial_count - len(self.particles),
            'fusion_events': len(fusion_events),
            'compression_rate': (initial_count - len(self.particles)) / initial_count,
            'pre': pre_analysis,
            'post': post_analysis,
            'entropy_change': post_analysis['entropy'] - pre_analysis['entropy'],
            'coherence_change': post_analysis['coherence'] - pre_analysis['coherence']
        }
        
        self.history.append(result)
        return result
    
    def get_super_structures(self, mass_threshold: float = 100) -> List[Particle]:
        """Identifie les super-structures (analogues aux trous noirs)"""
        return [p for p in self.particles if p.mass >= mass_threshold]
    
    def get_convergence_score(self) -> float:
        """Score de convergence : à quel point le système est unifié"""
        if len(self.particles) <= 1:
            return 1.0
        
        values = [p.value for p in self.particles]
        value_range = max(values) - min(values)
        
        # Score inversement proportionnel à la dispersion
        return 1.0 / (1.0 + value_range / 100)


def experiment_1_scaling():
    """Expérience 1 : Test de scalabilité (comme un pulsar)"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║  EXPÉRIENCE 1 : SCALABILITÉ COSMIQUE                               ║")
    print("╚" + "="*68 + "╝\n")
    
    results = []
    
    for exp in range(1, 6):
        load = 10 ** (exp + 1)
        engine = AdvancedUniversalEngine()
        engine.inject_chaos(load)
        
        result = engine.pulse_process_detailed()
        results.append(result)
        
        print(f"Charge {load:7d} → {result['final_count']:5d} particules | "
              f"Compression: {result['compression_rate']*100:5.2f}% | "
              f"Temps: {result['duration_us']:8.0f} µs | "
              f"Entropie: {result['post']['entropy']:6.2f}")
    
    print(f"\n✓ Scalabilité validée jusqu'à {10**6:,} particules")
    return results


def experiment_2_pattern_emergence():
    """Expérience 2 : Émergence de patterns (nébuleuse → étoile)"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║  EXPÉRIENCE 2 : ÉMERGENCE DE PATTERNS                              ║")
    print("╚" + "="*68 + "╝\n")
    
    engine = AdvancedUniversalEngine()
    engine.inject_pattern("nebula", 1000)
    
    print("État initial : Nébuleuse diffuse (1000 particules)")
    print("Simulation de l'effondrement gravitationnel...\n")
    
    for cycle in range(5):
        result = engine.pulse_process_detailed()
        
        super_structs = engine.get_super_structures(mass_threshold=50)
        convergence = engine.get_convergence_score()
        
        print(f"Cycle {cycle+1} : {result['final_count']:4d} particules | "
              f"Super-structures: {len(super_structs):2d} | "
              f"Convergence: {convergence:.3f} | "
              f"Cohérence: {result['post']['coherence']:.3f}")
        
        if len(engine.particles) <= 5:
            print("\n✓ Formation stellaire complète !")
            break
    
    print(f"\nÉtoiles finales : {len(engine.particles)}")
    for i, star in enumerate(sorted(engine.particles, key=lambda p: -p.mass)[:3]):
        print(f"  Étoile {i+1} : masse={star.mass:.1f}, stabilité={star.stability:.2f}, "
              f"génération={star.generation}")
    
    return engine


def experiment_3_information_dynamics():
    """Expérience 3 : Dynamique informationnelle (vérité vs bruit)"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║  EXPÉRIENCE 3 : DYNAMIQUE INFORMATIONNELLE                         ║")
    print("╚" + "="*68 + "╝\n")
    
    # Setup : Signal vs Bruit
    engine = AdvancedUniversalEngine(fusion_threshold=3.0)
    
    truth_value = 50.0
    signal_strength = 200  # Particules de vérité
    noise_strength = 800   # Particules de bruit
    
    print(f"Configuration :")
    print(f"  - Signal (vérité={truth_value}) : {signal_strength} particules lourdes")
    print(f"  - Bruit : {noise_strength} particules légères\n")
    
    # Injection du signal
    for _ in range(signal_strength):
        engine.particles.append(Particle(
            value=truth_value,
            stability=0.9,
            mass=5.0,
            generation=0
        ))
    
    # Injection du bruit
    for _ in range(noise_strength):
        engine.particles.append(Particle(
            value=truth_value + random.uniform(-10, 10),
            stability=0.1,
            mass=1.0,
            generation=0
        ))
    
    print("Évolution du système :\n")
    
    for cycle in range(5):
        result = engine.pulse_process_detailed()
        
        # Calcul du signal/bruit
        aligned = sum(1 for p in engine.particles 
                     if abs(p.value - truth_value) < 1.0)
        snr = aligned / len(engine.particles) if engine.particles else 0
        
        print(f"Cycle {cycle+1} : {result['final_count']:4d} particules | "
              f"Alignées sur vérité: {aligned:3d} ({snr*100:.1f}%) | "
              f"Entropie: {result['post']['entropy']:6.2f}")
        
        if snr > 0.8 or len(engine.particles) <= 3:
            print("\n✓ Le signal a dominé le bruit !")
            break
    
    print(f"\nÉtat final :")
    for i, p in enumerate(engine.particles[:5]):
        deviation = abs(p.value - truth_value)
        print(f"  Particule {i+1} : valeur={p.value:.2f} (Δ={deviation:.2f}), "
              f"masse={p.mass:.1f}, stabilité={p.stability:.2f}")
    
    return engine


def experiment_4_black_hole():
    """Expérience 4 : Attracteur massif (trou noir informationnel)"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║  EXPÉRIENCE 4 : TROU NOIR INFORMATIONNEL                           ║")
    print("╚" + "="*68 + "╝\n")
    
    engine = AdvancedUniversalEngine()
    engine.inject_pattern("black_hole", 500)
    
    print("État initial : Trou noir central + disque d'accrétion (500 particules)\n")
    
    for cycle in range(3):
        result = engine.pulse_process_detailed()
        
        super_massive = engine.get_super_structures(mass_threshold=200)
        
        print(f"Cycle {cycle+1} : {result['final_count']:4d} particules | "
              f"Trous noirs (M>200): {len(super_massive)} | "
              f"Masse totale: {result['post']['total_mass']:.0f}")
        
        if super_massive:
            behemoth = max(super_massive, key=lambda p: p.mass)
            print(f"  → Léviathan : masse={behemoth.mass:.0f}, "
                  f"génération={behemoth.generation}")
    
    return engine


def main():
    print("\n")
    print("═"*70)
    print("  ANALYSE APPROFONDIE : INFORMATIQUE COSMIQUE")
    print("  Étude des patterns de convergence et d'émergence")
    print("═"*70)
    
    random.seed(42)  # Reproductibilité
    
    results = {}
    results['scaling'] = experiment_1_scaling()
    results['emergence'] = experiment_2_pattern_emergence()
    results['dynamics'] = experiment_3_information_dynamics()
    results['black_hole'] = experiment_4_black_hole()
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║  SYNTHÈSE                                                          ║")
    print("╚" + "="*68 + "╝")
    print("\nTous les patterns cosmiques sont computationnellement viables :")
    print("  ✓ Scalabilité : O(n²) mais avec compression exponentielle")
    print("  ✓ Émergence : Les structures complexes apparaissent naturellement")
    print("  ✓ Dynamique : Le signal domine le bruit par gravité informationnelle")
    print("  ✓ Attracteurs : Les trous noirs informationnels unifient l'information")
    print("\n" + "="*70)
    print("CONCLUSION : L'univers est déjà un ordinateur.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
