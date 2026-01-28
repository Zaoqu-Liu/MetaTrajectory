"""
Metabolic Knowledge Base Module

Provides metabolic pathway definitions, network structure, and rate-limiting
enzyme information for metabolic trajectory analysis.

This module includes:
- Pre-defined metabolic pathways from KEGG (85 pathways from scMetabolism)
- Metabolic network graph construction
- Rate-limiting enzyme definitions
- Pathway-gene mappings

Data sources:
- KEGG pathways: scMetabolism package (Wu et al., Cancer Discovery 2021)
- REACTOME pathways: MSigDB
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import warnings


def parse_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
    """
    Parse a GMT file into a dictionary of pathways.
    
    GMT format: pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
    
    Parameters
    ----------
    gmt_path : str
        Path to GMT file
        
    Returns
    -------
    dict
        Pathway name -> list of genes
    """
    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                name = parts[0]
                genes = parts[2:]  # Skip description
                pathways[name] = genes
    return pathways


# Pathways to exclude from analysis (biologically inappropriate for scRNA-seq)
EXCLUDED_PATHWAYS = {
    # Bacterial pathways (humans don't synthesize antibiotics)
    "Neomycin, kanamycin and gentamicin biosynthesis",
    
    # Humans cannot synthesize essential amino acids de novo
    "Valine, leucine and isoleucine biosynthesis",
    "Phenylalanine, tyrosine and tryptophan biosynthesis",
    
    # D-amino acid metabolism is very limited in mammals
    "D-Arginine and D-ornithine metabolism",
    "D-Glutamine and D-glutamate metabolism",
    
    # Caffeine is exogenous, only liver-specific catabolism
    "Caffeine metabolism",
}

# Pathways with too few genes for reliable analysis (< 5 genes)
MIN_PATHWAY_GENES = 5


def filter_pathways(
    pathways: Dict[str, List[str]], 
    exclude_problematic: bool = True,
    min_genes: int = MIN_PATHWAY_GENES,
    exclude_drug_metabolism: bool = False,
    exclude_glycan_biosynthesis: bool = False
) -> Dict[str, List[str]]:
    """
    Filter pathways based on scientific criteria.
    
    Parameters
    ----------
    pathways : dict
        Pathway name -> gene list
    exclude_problematic : bool
        Remove biologically inappropriate pathways
    min_genes : int
        Minimum genes required per pathway
    exclude_drug_metabolism : bool
        Remove drug/xenobiotic metabolism (liver-specific)
    exclude_glycan_biosynthesis : bool
        Remove glycan biosynthesis (post-translational, not energy metabolism)
        
    Returns
    -------
    dict
        Filtered pathways
    """
    filtered = {}
    
    for name, genes in pathways.items():
        # Skip excluded pathways
        if exclude_problematic and name in EXCLUDED_PATHWAYS:
            continue
            
        # Skip pathways with too few genes
        if len(genes) < min_genes:
            continue
            
        # Optional: skip drug metabolism pathways
        if exclude_drug_metabolism:
            if "drug" in name.lower() or "xenobiotics" in name.lower():
                continue
        
        # Optional: skip glycan biosynthesis
        if exclude_glycan_biosynthesis:
            if "glycan" in name.lower() or "glycosyl" in name.lower():
                continue
        
        filtered[name] = genes
    
    return filtered


class MetabolicKnowledgeBase:
    """
    Metabolic knowledge base containing pathway definitions and network structure.
    
    This class provides:
    1. Curated metabolic pathway gene sets
    2. Rate-limiting enzyme definitions
    3. Metabolic network graph for distance calculation
    4. Metabolic axis definitions for biological interpretation
    
    Attributes
    ----------
    pathways : dict
        Pathway name -> list of genes
    rate_limiting_enzymes : set
        Set of rate-limiting enzyme genes
    network : networkx.DiGraph
        Metabolic network graph
    metabolic_axes : dict
        Metabolic axis definitions (e.g., glycolysis vs OXPHOS)
        
    Examples
    --------
    >>> kb = MetabolicKnowledgeBase()
    >>> kb.load_kegg_pathways()
    >>> print(kb.get_pathway_genes("Glycolysis"))
    ['HK1', 'HK2', 'GPI', 'PFKM', ...]
    """
    
    def __init__(self):
        """Initialize empty knowledge base."""
        self.pathways: Dict[str, List[str]] = {}
        self.rate_limiting_enzymes: Set[str] = set()
        self.network: Optional[nx.DiGraph] = None
        self.metabolic_axes: Dict[str, Dict[str, List[str]]] = {}
        self._initialized = False
    
    def load_kegg_pathways(
        self, 
        species: str = "hsa",
        gmt_file: Optional[str] = None,
        use_full_kegg: bool = True,
        filter_problematic: bool = True,
        min_genes: int = 5,
        exclude_drug_metabolism: bool = False,
        exclude_glycan_biosynthesis: bool = False
    ) -> "MetabolicKnowledgeBase":
        """
        Load KEGG metabolic pathways with scientific filtering.
        
        Can load from:
        1. Full KEGG GMT file (85 pathways from scMetabolism) - default
        2. Core pathways only (17 curated pathways)
        3. Custom GMT file
        
        Parameters
        ----------
        species : str
            Species code (currently only 'hsa' for human is supported)
        gmt_file : str, optional
            Path to custom GMT file. If provided, loads from this file.
        use_full_kegg : bool
            If True (default), loads KEGG pathways from GMT.
            If False, uses 17 core curated pathways.
        filter_problematic : bool
            If True (default), removes biologically inappropriate pathways:
            - Bacterial pathways (antibiotic synthesis)
            - Essential amino acid biosynthesis (humans can't do this)
            - D-amino acid metabolism (very limited in mammals)
        min_genes : int
            Minimum genes required per pathway (default 5)
        exclude_drug_metabolism : bool
            If True, removes drug/xenobiotic metabolism pathways (liver-specific)
        exclude_glycan_biosynthesis : bool
            If True, removes glycan biosynthesis pathways (post-translational)
            
        Returns
        -------
        self
            Returns self for method chaining
            
        Notes
        -----
        After filtering with default settings, typically ~75 pathways remain.
        
        Excluded by default:
        - Neomycin, kanamycin and gentamicin biosynthesis (bacterial)
        - Valine, leucine and isoleucine biosynthesis (essential AA)
        - Phenylalanine, tyrosine and tryptophan biosynthesis (essential AA)
        - D-Arginine and D-ornithine metabolism (1 gene only)
        - D-Glutamine and D-glutamate metabolism
        - Caffeine metabolism
        - Biotin metabolism, Lipoic acid metabolism (< 5 genes)
        """
        if species != "hsa":
            warnings.warn(f"Species '{species}' not fully supported, using human pathways")
        
        # Try to load from GMT file
        if gmt_file is not None:
            pathways = parse_gmt_file(gmt_file)
            self.pathways = filter_pathways(
                pathways, 
                exclude_problematic=filter_problematic,
                min_genes=min_genes,
                exclude_drug_metabolism=exclude_drug_metabolism,
                exclude_glycan_biosynthesis=exclude_glycan_biosynthesis
            )
            self._load_rate_limiting_enzymes()
            self._load_metabolic_axes()
            self._build_network()
            self._initialized = True
            return self
        
        # Try to load full KEGG from bundled GMT
        if use_full_kegg:
            # Find the GMT file in package data
            package_dir = Path(__file__).parent.parent.parent
            gmt_path = package_dir / "data" / "KEGG_metabolism_nc.gmt"
            
            if gmt_path.exists():
                pathways = parse_gmt_file(str(gmt_path))
                self.pathways = filter_pathways(
                    pathways, 
                    exclude_problematic=filter_problematic,
                    min_genes=min_genes,
                    exclude_drug_metabolism=exclude_drug_metabolism,
                    exclude_glycan_biosynthesis=exclude_glycan_biosynthesis
                )
                self._load_rate_limiting_enzymes()
                self._load_metabolic_axes()
                self._build_network()
                self._initialized = True
                return self
            else:
                warnings.warn(
                    f"KEGG GMT file not found at {gmt_path}. "
                    "Falling back to core pathways."
                )
        
        # Core metabolic pathways with gene symbols
        self.pathways = {
            # Carbohydrate metabolism
            "Glycolysis": [
                "HK1", "HK2", "HK3", "GCK",  # Hexokinase
                "GPI",  # Glucose-6-phosphate isomerase
                "PFKM", "PFKL", "PFKP",  # Phosphofructokinase
                "ALDOA", "ALDOB", "ALDOC",  # Aldolase
                "TPI1",  # Triosephosphate isomerase
                "GAPDH",  # Glyceraldehyde 3-phosphate dehydrogenase
                "PGK1", "PGK2",  # Phosphoglycerate kinase
                "PGAM1", "PGAM2",  # Phosphoglycerate mutase
                "ENO1", "ENO2", "ENO3",  # Enolase
                "PKM", "PKLR",  # Pyruvate kinase
            ],
            
            "Gluconeogenesis": [
                "PC",  # Pyruvate carboxylase
                "PCK1", "PCK2",  # Phosphoenolpyruvate carboxykinase
                "FBP1", "FBP2",  # Fructose-1,6-bisphosphatase
                "G6PC", "G6PC2", "G6PC3",  # Glucose-6-phosphatase
            ],
            
            "TCA_Cycle": [
                "CS",  # Citrate synthase
                "ACO1", "ACO2",  # Aconitase
                "IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G",  # Isocitrate dehydrogenase
                "OGDH", "OGDHL",  # Oxoglutarate dehydrogenase
                "SUCLA2", "SUCLG1", "SUCLG2",  # Succinyl-CoA synthetase
                "SDHA", "SDHB", "SDHC", "SDHD",  # Succinate dehydrogenase
                "FH",  # Fumarate hydratase
                "MDH1", "MDH2",  # Malate dehydrogenase
            ],
            
            "Pentose_Phosphate": [
                "G6PD",  # Glucose-6-phosphate dehydrogenase
                "PGLS",  # 6-phosphogluconolactonase
                "PGD",  # 6-phosphogluconate dehydrogenase
                "RPE",  # Ribulose-5-phosphate-3-epimerase
                "RPIA",  # Ribose-5-phosphate isomerase
                "TKT",  # Transketolase
                "TALDO1",  # Transaldolase
            ],
            
            # Oxidative phosphorylation
            "OXPHOS": [
                # Complex I (NADH dehydrogenase)
                "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS4", "NDUFS5", "NDUFS6", "NDUFS7", "NDUFS8",
                "NDUFV1", "NDUFV2", "NDUFV3",
                "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5", "NDUFA6", "NDUFA7", "NDUFA8", "NDUFA9", "NDUFA10", "NDUFA11", "NDUFA12", "NDUFA13",
                "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5", "NDUFB6", "NDUFB7", "NDUFB8", "NDUFB9", "NDUFB10", "NDUFB11",
                "NDUFC1", "NDUFC2",
                # Complex II (Succinate dehydrogenase) - also in TCA
                "SDHA", "SDHB", "SDHC", "SDHD",
                # Complex III (Cytochrome bc1)
                "UQCRC1", "UQCRC2", "UQCRFS1", "UQCRB", "UQCRH", "UQCRQ", "UQCR10", "UQCR11",
                "CYC1",  # Cytochrome c1
                # Complex IV (Cytochrome c oxidase)
                "COX4I1", "COX4I2", "COX5A", "COX5B", "COX6A1", "COX6A2", "COX6B1", "COX6B2", "COX6C",
                "COX7A1", "COX7A2", "COX7B", "COX7C", "COX8A", "COX8C",
                "MT-CO1", "MT-CO2", "MT-CO3",
                # Complex V (ATP synthase)
                "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E",
                "ATP5PB", "ATP5MC1", "ATP5MC2", "ATP5MC3", "ATP5PD", "ATP5ME", "ATP5PF", "ATP5MG", "ATP5MPL", "ATP5IF1",
            ],
            
            # Lipid metabolism
            "Fatty_Acid_Synthesis": [
                "ACACA", "ACACB",  # Acetyl-CoA carboxylase
                "FASN",  # Fatty acid synthase
                "ACLY",  # ATP citrate lyase
                "SCD",  # Stearoyl-CoA desaturase
                "ELOVL1", "ELOVL2", "ELOVL3", "ELOVL4", "ELOVL5", "ELOVL6", "ELOVL7",  # Elongation
                "FADS1", "FADS2",  # Fatty acid desaturase
            ],
            
            "Fatty_Acid_Oxidation": [
                "CPT1A", "CPT1B", "CPT1C",  # Carnitine palmitoyltransferase 1
                "CPT2",  # Carnitine palmitoyltransferase 2
                "SLC25A20",  # Carnitine-acylcarnitine translocase
                "ACADVL", "ACADL", "ACADM", "ACADS",  # Acyl-CoA dehydrogenase
                "HADHA", "HADHB",  # Trifunctional protein
                "ECHS1",  # Enoyl-CoA hydratase
                "HADH",  # Hydroxyacyl-CoA dehydrogenase
                "ACAA1", "ACAA2",  # Acetyl-CoA acyltransferase
            ],
            
            "Cholesterol_Synthesis": [
                "HMGCS1", "HMGCS2",  # HMG-CoA synthase
                "HMGCR",  # HMG-CoA reductase (rate-limiting!)
                "MVK",  # Mevalonate kinase
                "PMVK",  # Phosphomevalonate kinase
                "MVD",  # Mevalonate diphosphate decarboxylase
                "IDI1", "IDI2",  # Isopentenyl-diphosphate isomerase
                "FDPS",  # Farnesyl diphosphate synthase
                "FDFT1",  # Squalene synthase
                "SQLE",  # Squalene epoxidase
                "LSS",  # Lanosterol synthase
                "DHCR7", "DHCR24",  # 7-dehydrocholesterol reductase
            ],
            
            # Amino acid metabolism
            "Glutamine_Metabolism": [
                "GLS", "GLS2",  # Glutaminase
                "GLUL",  # Glutamine synthetase
                "GLUD1", "GLUD2",  # Glutamate dehydrogenase
                "GOT1", "GOT2",  # Glutamic-oxaloacetic transaminase
                "GPT", "GPT2",  # Glutamic-pyruvic transaminase
                "SLC1A5", "SLC38A1", "SLC38A2",  # Glutamine transporters
            ],
            
            "Serine_Glycine_Metabolism": [
                "PHGDH",  # Phosphoglycerate dehydrogenase
                "PSAT1",  # Phosphoserine aminotransferase
                "PSPH",  # Phosphoserine phosphatase
                "SHMT1", "SHMT2",  # Serine hydroxymethyltransferase
                "MTHFD1", "MTHFD2", "MTHFD1L", "MTHFD2L",  # Methylenetetrahydrofolate dehydrogenase
            ],
            
            "Branched_Chain_AA": [
                "BCAT1", "BCAT2",  # Branched-chain amino acid transaminase
                "BCKDHA", "BCKDHB",  # Branched-chain keto acid dehydrogenase
                "DBT",  # Dihydrolipoamide branched chain transacylase
                "DLD",  # Dihydrolipoamide dehydrogenase
            ],
            
            # One-carbon metabolism
            "One_Carbon_Metabolism": [
                "MTHFR",  # Methylenetetrahydrofolate reductase
                "MTR",  # Methionine synthase
                "MTRR",  # Methionine synthase reductase
                "MAT1A", "MAT2A", "MAT2B",  # Methionine adenosyltransferase
                "AHCY",  # Adenosylhomocysteinase
                "CBS",  # Cystathionine beta-synthase
                "CTH",  # Cystathionine gamma-lyase
            ],
            
            # Nucleotide metabolism
            "Purine_Synthesis": [
                "PPAT",  # Phosphoribosyl pyrophosphate amidotransferase
                "GART",  # Glycinamide ribonucleotide transformylase
                "PFAS",  # Phosphoribosylformylglycinamidine synthase
                "PAICS",  # Phosphoribosylaminoimidazole carboxylase
                "ADSL",  # Adenylosuccinate lyase
                "ATIC",  # AICAR transformylase
                "IMPDH1", "IMPDH2",  # IMP dehydrogenase
                "GMPS",  # GMP synthase
            ],
            
            "Pyrimidine_Synthesis": [
                "CAD",  # Carbamoyl-phosphate synthetase 2
                "DHODH",  # Dihydroorotate dehydrogenase
                "UMPS",  # UMP synthase
                "UCK1", "UCK2",  # Uridine-cytidine kinase
                "CMPK1", "CMPK2",  # Cytidine monophosphate kinase
                "NME1", "NME2",  # Nucleoside diphosphate kinase
                "RRM1", "RRM2",  # Ribonucleotide reductase
                "TYMS",  # Thymidylate synthase
            ],
            
            # Hypoxia and lactate
            "Hypoxia_Metabolism": [
                "LDHA", "LDHB", "LDHC",  # Lactate dehydrogenase
                "SLC16A1", "SLC16A3",  # Monocarboxylate transporters (MCT1, MCT4)
                "PDK1", "PDK2", "PDK3", "PDK4",  # Pyruvate dehydrogenase kinase
                "PDHA1", "PDHB",  # Pyruvate dehydrogenase
                "HIF1A",  # Hypoxia-inducible factor 1 alpha
                "EPAS1",  # HIF2A
                "VEGFA",  # Vascular endothelial growth factor
                "SLC2A1", "SLC2A3",  # Glucose transporters (GLUT1, GLUT3)
            ],
            
            # Ketone body metabolism
            "Ketogenesis": [
                "HMGCS2",  # HMG-CoA synthase 2 (mitochondrial)
                "HMGCL",  # HMG-CoA lyase
                "BDH1", "BDH2",  # 3-hydroxybutyrate dehydrogenase
                "OXCT1", "OXCT2",  # Succinyl-CoA:3-ketoacid-CoA transferase
            ],
            
            # Reactive oxygen species
            "Antioxidant": [
                "SOD1", "SOD2", "SOD3",  # Superoxide dismutase
                "CAT",  # Catalase
                "GPX1", "GPX2", "GPX3", "GPX4",  # Glutathione peroxidase
                "GSR",  # Glutathione reductase
                "GCLC", "GCLM",  # Glutamate-cysteine ligase
                "GSS",  # Glutathione synthetase
                "TXN", "TXN2",  # Thioredoxin
                "TXNRD1", "TXNRD2",  # Thioredoxin reductase
                "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6",  # Peroxiredoxin
                "NQO1",  # NAD(P)H quinone dehydrogenase
            ],
        }
        
        self._load_rate_limiting_enzymes()
        self._load_metabolic_axes()
        self._build_network()
        self._initialized = True
        
        return self
    
    def _load_rate_limiting_enzymes(self) -> None:
        """Load curated list of rate-limiting enzymes."""
        self.rate_limiting_enzymes = {
            # Glycolysis
            "HK1", "HK2", "HK3", "GCK",  # Hexokinase
            "PFKM", "PFKL", "PFKP",  # Phosphofructokinase (committed step)
            "PKM", "PKLR",  # Pyruvate kinase
            
            # Gluconeogenesis
            "PC", "PCK1", "PCK2",  # PEPCK
            "FBP1", "FBP2",  # Fructose-1,6-bisphosphatase
            "G6PC",  # Glucose-6-phosphatase
            
            # TCA cycle
            "CS",  # Citrate synthase
            "IDH1", "IDH2", "IDH3A",  # Isocitrate dehydrogenase
            "OGDH",  # Alpha-ketoglutarate dehydrogenase
            
            # OXPHOS
            "NDUFS1",  # Complex I
            "SDHA",  # Complex II
            "UQCRC1",  # Complex III
            "COX4I1",  # Complex IV
            "ATP5F1A",  # Complex V
            
            # Fatty acid metabolism
            "ACACA", "ACACB",  # ACC (fatty acid synthesis)
            "FASN",  # Fatty acid synthase
            "CPT1A", "CPT1B",  # CPT1 (fatty acid oxidation)
            
            # Cholesterol
            "HMGCR",  # HMG-CoA reductase (statin target!)
            
            # Amino acid
            "GLS", "GLS2",  # Glutaminase
            "GLUL",  # Glutamine synthetase
            "PHGDH",  # Serine synthesis
            
            # Nucleotide
            "RRM1", "RRM2",  # Ribonucleotide reductase
            "IMPDH1", "IMPDH2",  # IMP dehydrogenase
            
            # Hypoxia
            "LDHA",  # Lactate dehydrogenase A
            "PDK1",  # Pyruvate dehydrogenase kinase
        }
    
    def _load_metabolic_axes(self) -> None:
        """
        Define metabolic axes for biological interpretation.
        
        Each axis represents a metabolic trade-off or transition:
        - positive: genes associated with one end of the axis
        - negative: genes associated with the other end
        """
        self.metabolic_axes = {
            "Glycolysis_vs_OXPHOS": {
                "positive": ["HK1", "HK2", "PFKM", "PFKL", "PKM", "LDHA", "PDK1"],
                "negative": ["CS", "IDH2", "SDHA", "ATP5F1A", "COX4I1", "NDUFS1"],
                "description": "Warburg effect: glycolytic vs oxidative metabolism"
            },
            "Anabolism_vs_Catabolism": {
                "positive": ["ACACA", "FASN", "ACLY", "PHGDH", "PSAT1"],
                "negative": ["CPT1A", "GLS", "ACADM", "HADHA"],
                "description": "Biosynthesis vs breakdown"
            },
            "Proliferative_vs_Quiescent": {
                "positive": ["PKM", "LDHA", "GLS", "PHGDH", "RRM1", "RRM2", "TYMS"],
                "negative": ["IDH2", "CS", "CPT1A", "ACADM"],
                "description": "High proliferation vs low proliferation metabolic profile"
            },
            "Redox_Stress": {
                "positive": ["G6PD", "PGD", "GSR", "GCLC", "NQO1"],
                "negative": ["SOD1", "SOD2", "CAT"],
                "description": "Antioxidant response activation"
            },
            "Lipid_Synthesis_vs_Oxidation": {
                "positive": ["ACACA", "FASN", "SCD", "ACLY"],
                "negative": ["CPT1A", "CPT1B", "ACADM", "HADHA"],
                "description": "Lipogenesis vs lipolysis"
            }
        }
    
    def _build_network(self) -> None:
        """
        Build metabolic network graph from pathway connections.
        
        The network represents metabolic relationships where nodes are
        pathways and edges represent shared metabolites or regulatory connections.
        """
        self.network = nx.DiGraph()
        
        # Add pathways as nodes
        for pathway in self.pathways:
            self.network.add_node(pathway, genes=self.pathways[pathway])
        
        # Define pathway connections (based on metabolic logic)
        connections = [
            # Central carbon metabolism
            ("Glycolysis", "TCA_Cycle", 1.0),  # Pyruvate -> Acetyl-CoA
            ("Glycolysis", "Pentose_Phosphate", 0.8),  # G6P branch
            ("Glycolysis", "Hypoxia_Metabolism", 0.9),  # Pyruvate -> Lactate
            ("Gluconeogenesis", "Glycolysis", 0.7),  # Reverse reactions
            ("TCA_Cycle", "OXPHOS", 1.0),  # NADH/FADH2 production
            ("TCA_Cycle", "Gluconeogenesis", 0.6),  # OAA -> PEP
            
            # Lipid metabolism
            ("Glycolysis", "Fatty_Acid_Synthesis", 0.7),  # Acetyl-CoA source
            ("TCA_Cycle", "Fatty_Acid_Synthesis", 0.8),  # Citrate export
            ("Fatty_Acid_Oxidation", "TCA_Cycle", 0.9),  # Acetyl-CoA
            ("Fatty_Acid_Oxidation", "Ketogenesis", 0.7),  # Ketone bodies
            ("Cholesterol_Synthesis", "Fatty_Acid_Synthesis", 0.5),  # Shared precursors
            
            # Amino acid metabolism
            ("TCA_Cycle", "Glutamine_Metabolism", 0.8),  # α-KG
            ("Glutamine_Metabolism", "TCA_Cycle", 0.8),  # Anaplerosis
            ("Glycolysis", "Serine_Glycine_Metabolism", 0.7),  # 3-PG branch
            ("Serine_Glycine_Metabolism", "One_Carbon_Metabolism", 0.9),  # Folate cycle
            ("Branched_Chain_AA", "TCA_Cycle", 0.6),  # Succinyl-CoA
            
            # Nucleotide metabolism
            ("Pentose_Phosphate", "Purine_Synthesis", 0.8),  # R5P
            ("Pentose_Phosphate", "Pyrimidine_Synthesis", 0.8),  # R5P
            ("One_Carbon_Metabolism", "Purine_Synthesis", 0.7),  # Folate
            ("One_Carbon_Metabolism", "Pyrimidine_Synthesis", 0.7),  # Folate
            ("Glutamine_Metabolism", "Purine_Synthesis", 0.6),  # Nitrogen donor
            ("Glutamine_Metabolism", "Pyrimidine_Synthesis", 0.6),  # Nitrogen donor
            
            # Redox and stress
            ("Pentose_Phosphate", "Antioxidant", 0.7),  # NADPH
            ("OXPHOS", "Antioxidant", 0.5),  # ROS production -> defense
            
            # Hypoxia connections
            ("Hypoxia_Metabolism", "Glycolysis", 0.8),  # HIF1A upregulates glycolysis
        ]
        
        for source, target, weight in connections:
            if source in self.network.nodes and target in self.network.nodes:
                self.network.add_edge(source, target, weight=weight)
    
    def get_pathway_genes(self, pathway: str) -> List[str]:
        """
        Get genes for a specific pathway.
        
        Parameters
        ----------
        pathway : str
            Pathway name
            
        Returns
        -------
        list of str
            Gene symbols in the pathway
        """
        if pathway not in self.pathways:
            raise ValueError(f"Pathway '{pathway}' not found. Available: {list(self.pathways.keys())}")
        return self.pathways[pathway]
    
    def get_all_genes(self) -> List[str]:
        """Get all unique genes across all pathways."""
        all_genes = set()
        for genes in self.pathways.values():
            all_genes.update(genes)
        return sorted(list(all_genes))
    
    def get_rate_limiting_genes(self) -> List[str]:
        """Get list of rate-limiting enzyme genes."""
        return sorted(list(self.rate_limiting_enzymes))
    
    def get_pathway_for_gene(self, gene: str) -> List[str]:
        """Get pathways containing a specific gene."""
        pathways = []
        for pathway, genes in self.pathways.items():
            if gene in genes:
                pathways.append(pathway)
        return pathways
    
    def get_metabolic_axis(self, axis_name: str) -> Dict[str, List[str]]:
        """
        Get gene sets for a metabolic axis.
        
        Parameters
        ----------
        axis_name : str
            Name of the metabolic axis
            
        Returns
        -------
        dict
            Dictionary with 'positive' and 'negative' gene lists
        """
        if axis_name not in self.metabolic_axes:
            raise ValueError(f"Axis '{axis_name}' not found. Available: {list(self.metabolic_axes.keys())}")
        return self.metabolic_axes[axis_name]
    
    def compute_pathway_distance(
        self,
        pathway1: str,
        pathway2: str,
        method: str = "shortest_path"
    ) -> float:
        """
        Compute distance between two pathways in the metabolic network.
        
        Parameters
        ----------
        pathway1 : str
            First pathway name
        pathway2 : str
            Second pathway name
        method : str
            Distance method: 'shortest_path' or 'jaccard'
            
        Returns
        -------
        float
            Distance between pathways
        """
        if method == "shortest_path":
            if self.network is None:
                raise ValueError("Network not built. Call load_kegg_pathways() first.")
            
            try:
                # Shortest path in undirected version
                G_undirected = self.network.to_undirected()
                path_length = nx.shortest_path_length(G_undirected, pathway1, pathway2)
                return float(path_length)
            except nx.NetworkXNoPath:
                return float('inf')
                
        elif method == "jaccard":
            genes1 = set(self.pathways.get(pathway1, []))
            genes2 = set(self.pathways.get(pathway2, []))
            
            if len(genes1) == 0 or len(genes2) == 0:
                return 1.0
            
            intersection = len(genes1 & genes2)
            union = len(genes1 | genes2)
            
            return 1.0 - (intersection / union)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_pathway_distance_matrix(self, method: str = "shortest_path") -> pd.DataFrame:
        """
        Compute pairwise distance matrix between all pathways.
        
        Parameters
        ----------
        method : str
            Distance method
            
        Returns
        -------
        pd.DataFrame
            Symmetric distance matrix
        """
        pathways = list(self.pathways.keys())
        n = len(pathways)
        
        distances = np.zeros((n, n))
        for i, p1 in enumerate(pathways):
            for j, p2 in enumerate(pathways):
                if i <= j:
                    d = self.compute_pathway_distance(p1, p2, method)
                    distances[i, j] = d
                    distances[j, i] = d
        
        return pd.DataFrame(distances, index=pathways, columns=pathways)
    
    def filter_genes_by_expression(
        self,
        adata,
        min_cells: int = 10,
        min_counts: float = 0.0
    ) -> "MetabolicKnowledgeBase":
        """
        Filter pathway genes based on expression in the dataset.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        min_cells : int
            Minimum number of cells expressing the gene
        min_counts : float
            Minimum mean expression
            
        Returns
        -------
        self
            Returns self with filtered pathways
        """
        import warnings
        
        available_genes = set(adata.var_names)
        
        filtered_pathways = {}
        for pathway, genes in self.pathways.items():
            # Filter to available genes
            pathway_genes = [g for g in genes if g in available_genes]
            
            if len(pathway_genes) >= 3:  # Require at least 3 genes
                filtered_pathways[pathway] = pathway_genes
            else:
                warnings.warn(f"Pathway '{pathway}' has <3 detected genes, skipping")
        
        self.pathways = filtered_pathways
        
        # Also filter rate-limiting enzymes
        self.rate_limiting_enzymes = self.rate_limiting_enzymes & available_genes
        
        return self
    
    def summary(self) -> str:
        """Return summary string of the knowledge base."""
        if not self._initialized:
            return "Knowledge base not initialized. Call load_kegg_pathways() first."
        
        n_pathways = len(self.pathways)
        n_genes = len(self.get_all_genes())
        n_rate_limiting = len(self.rate_limiting_enzymes)
        n_axes = len(self.metabolic_axes)
        n_edges = self.network.number_of_edges() if self.network else 0
        
        return (
            f"MetabolicKnowledgeBase Summary:\n"
            f"  - Pathways: {n_pathways}\n"
            f"  - Total genes: {n_genes}\n"
            f"  - Rate-limiting enzymes: {n_rate_limiting}\n"
            f"  - Metabolic axes: {n_axes}\n"
            f"  - Network edges: {n_edges}"
        )
    
    def __repr__(self) -> str:
        return self.summary()
