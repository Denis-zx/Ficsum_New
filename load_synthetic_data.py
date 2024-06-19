from skmultiflow.data.data_stream import DataStream
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator

import pandas as pd
import numpy as np


def RTREEGenerator(classification_function, random_state):
    return RandomTreeGenerator(tree_random_state=classification_function, sample_random_state=random_state)
def STAGGERGeneratorWrapper(classification_function, random_state):
    return STAGGERGenerator(classification_function=classification_function%3, random_state=random_state)
def SEAGeneratorWrapper(classification_function, random_state):
    return SEAGenerator(classification_function=classification_function%4, random_state=random_state)
def AGRAWALGeneratorWrapper(classification_function, random_state):
    return AGRAWALGenerator(classification_function=classification_function%10, random_state=random_state)

def RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def FeatureWeightExpGen(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: FeatureWeightExpGenerator(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: FeatureWeightExpGenerator(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def SigNoiseGen(signal_noise_ratio):
    return lambda classification_function, random_state: SigNoiseGenerator(tree_random_state=classification_function, sampler_random_state = random_state, signal_noise_ratio=signal_noise_ratio)
def RTREESAMPLEHPGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform'):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorHPFeatureSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, sampler_features = sampler_features, strength=1)
    return lambda classification_function, random_state: RandomTreeGeneratorHPFeatureSample(tree_random_state=classification_function, intra_concept_dist=intra_concept_dist, inter_concept_dist=inter_concept_dist, sampler_random_state = random_state, strength=1)
def HPLANESAMPLEGenerator(sampler_features):
    if sampler_features:
        return lambda classification_function, random_state: HyperplaneSampleGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0, sampler_random_state = random_state, sampler_features = sampler_features)
    return lambda classification_function, random_state: HyperplaneSampleGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0, sampler_random_state = random_state)

def HPLANEGenerator(classification_function, random_state):
    return HyperplaneGenerator(random_state=classification_function, n_features=8, n_drift_features=0, mag_change=0)
def RBFGenerator(classification_function, random_state):
    return RandomRBFGenerator(model_random_state=classification_function, sample_random_state=random_state)
def RBFGeneratorDifficulty(difficulty):
    n_centroids = difficulty * 5 + 15
    return lambda classification_function, random_state: RandomRBFGenerator(model_random_state=classification_function, sample_random_state=random_state, n_centroids=n_centroids, n_classes=4)
def RTREEGeneratorDifficulty(difficulty = 0):
    return lambda classification_function, random_state: RandomTreeGenerator(tree_random_state=classification_function, sample_random_state=random_state, max_tree_depth=difficulty+1, min_leaf_depth=difficulty)
def RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty = 0, strength = 1):
    if sampler_features:
        return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, sampler_random_state = random_state, sampler_features = sampler_features, max_tree_depth=difficulty+2, min_leaf_depth=difficulty, strength = strength)
    return lambda classification_function, random_state: RandomTreeGeneratorSample(tree_random_state=classification_function, sampler_random_state = random_state, max_tree_depth=difficulty+2, min_leaf_depth=difficulty, strength = strength)
    
    

def create_synthetic_concepts(path, name, seed):
    stream_generator = None
    num_concepts = None
    sampler_features = None
    if name == "STAGGER":
        stream_generator = STAGGERGeneratorWrapper
        num_concepts = 10
    if name == "STAGGERS":
        stream_generator = STAGGERGeneratorWrapper
        num_concepts = 3
    if name == "ARGWAL":
        stream_generator = AGRAWALGeneratorWrapper
        num_concepts = 10
    if name == "SEA":
        stream_generator = SEAGeneratorWrapper
        num_concepts = 10
    if name == "RTREE":
        stream_generator = RTREEGenerator
        num_concepts = 10
    if name == "RTREEEasy":
        stream_generator = RTREEGeneratorDifficulty(difficulty=0)
        num_concepts = 10
    if name == "RTREEEasySAMPLE":
        stream_generator = RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty=0, strength=0.1)
        num_concepts = 10
    if name == "RTREEMedSAMPLE":
        stream_generator = RTREESAMPLEGeneratorDifficulty(sampler_features, difficulty=0, strength=0.2)
        num_concepts = 10
    if name == "SynEasyF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyA":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['autocorrelation'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyD":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyAF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['autocorrelation', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDA":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'autocorrelation'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "SynEasyDAF":
        stream_generator = RTREESAMPLEGeneratorDifficulty(['distribution', 'autocorrelation', 'frequency'], difficulty=0)
        num_concepts = 10
    if name == "RTREESAMPLE":
        stream_generator = RTREESAMPLEGenerator(sampler_features)
        num_concepts = 10
    if name == "HPLANE":
        stream_generator = HPLANEGenerator
        num_concepts = 10
    if name == "RBF":
        stream_generator = RBFGenerator
        num_concepts = 10
    if name == "RBFEasy":
        stream_generator = RBFGeneratorDifficulty(difficulty=0)
        num_concepts = 10
    if name == "RBFMed":
        stream_generator = RBFGeneratorDifficulty(difficulty=2)
        num_concepts = 10
    if name == "HPLANESAMPLE":
        stream_generator = HPLANESAMPLEGenerator(sampler_features)
        num_concepts = 10
    if name == "RTREESAMPLE-UU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-UN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-UD":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-NU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-NN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-ND":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-DU":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='uniform')
        num_concepts = 10
    if name == "RTREESAMPLE-DN":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='norm')
        num_concepts = 10
    if name == "RTREESAMPLE-DD":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='dist')
        num_concepts = 10
    if name == "RTREESAMPLE-UB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='uniform', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLE-NB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='norm', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLE-DB":
        stream_generator = RTREESAMPLEGenerator(sampler_features, intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-A":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f1', 'f2', 'f3', 'f4'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-23":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f2', 'f3'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "RTREESAMPLEHP-14":
        stream_generator = RTREESAMPLEHPGenerator(sampler_features=['f1', 'f4'], intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "FeatureWeightExpGenerator":
        stream_generator = FeatureWeightExpGen(sampler_features, intra_concept_dist='dist', inter_concept_dist='bimodal')
        num_concepts = 10
    if name == "SigNoiseGenerator-1":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.1)
        num_concepts = 10
    if name == "SigNoiseGenerator-2":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.2)
        num_concepts = 10
    if name == "SigNoiseGenerator-3":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.3)
        num_concepts = 10
    if name == "SigNoiseGenerator-4":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.4)
        num_concepts = 10
    if name == "SigNoiseGenerator-5":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.5)
        num_concepts = 10
    if name == "SigNoiseGenerator-6":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.6)
        num_concepts = 10
    if name == "SigNoiseGenerator-7":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.7)
        num_concepts = 10
    if name == "SigNoiseGenerator-8":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.8)
        num_concepts = 10
    if name == "SigNoiseGenerator-9":
        stream_generator = SigNoiseGen(signal_noise_ratio=0.9)
        num_concepts = 10
    if name == "SigNoiseGenerator-10":
        stream_generator = SigNoiseGen(signal_noise_ratio=1.0)
        num_concepts = 10
    if stream_generator is None:
        raise ValueError("name not valid for a dataset")
    print(num_concepts)
    if seed is None:
        seed = np.random.randint(0, 1000)
    for c in range(num_concepts):
        concept = stream_generator(classification_function = seed + c, random_state=seed+c)
        with (path / f"concept_{c}.pickle").open("wb") as f:
            pickle.dump(concept, f)
        
        if hasattr(concept, 'get_data'):
            with (path / f"data_{c}.json").open("w") as f:
                json.dump(concept.get_data(), f)



def load_synthetic_concepts(name, seed, raw_data_path = None):
    data_path =  pathlib.Path(raw_data_path).resolve()

    file_path = data_path / name / "seeds" / str(seed)
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)
        create_synthetic_concepts(file_path, name, seed)

    concept_paths = list(file_path.glob('*concept*'))
    if len(concept_paths) == 0:
        create_synthetic_concepts(file_path, name, seed)
    concept_paths = list(file_path.glob('*concept*'))

    concepts = []
    for cp in concept_paths:
        with cp.open("rb") as f:
            concepts.append((pickle.load(f), cp.stem))
    concepts = sorted(concepts, key=lambda x: x[1])
    return concepts