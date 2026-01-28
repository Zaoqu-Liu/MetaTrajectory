#' @title Metabolic Knowledge Base
#' @description Build metabolic knowledge base with pathway definitions and network structure.
#' @name knowledge_base
NULL

#' Parse GMT File
#'
#' @description
#' Parse a GMT file into a list of pathways.
#'
#' @param gmt_path Path to GMT file.
#'
#' @return Named list of character vectors (pathway genes).
#'
#' @export
parse_gmt_file <- function(gmt_path) {
  lines <- readLines(gmt_path, warn = FALSE)
  pathways <- list()
  
  for (line in lines) {
    parts <- strsplit(line, "\t")[[1]]
    if (length(parts) >= 3) {
      name <- parts[1]
      genes <- parts[3:length(parts)]
      pathways[[name]] <- genes
    }
  }
  
  pathways
}

#' Biologically inappropriate pathways to exclude
#' @keywords internal
EXCLUDED_PATHWAYS <- c(
  # Bacterial pathways (humans don't synthesize antibiotics)
  "Neomycin, kanamycin and gentamicin biosynthesis",
  
  # Humans cannot synthesize essential amino acids de novo
  "Valine, leucine and isoleucine biosynthesis",
  "Phenylalanine, tyrosine and tryptophan biosynthesis",
  
  # D-amino acid metabolism is very limited in mammals
  "D-Arginine and D-ornithine metabolism",
  "D-Glutamine and D-glutamate metabolism",
  
  # Caffeine is exogenous, only liver-specific catabolism
  "Caffeine metabolism"
)

#' Filter Pathways
#'
#' @description
#' Filter pathways based on scientific criteria.
#'
#' @param pathways Named list of pathway gene vectors.
#' @param exclude_problematic Logical. Remove biologically inappropriate pathways.
#' @param min_genes Integer. Minimum genes required per pathway.
#' @param exclude_drug_metabolism Logical. Remove drug/xenobiotic metabolism.
#' @param exclude_glycan_biosynthesis Logical. Remove glycan biosynthesis.
#'
#' @return Filtered pathway list.
#'
#' @export
filter_pathways <- function(pathways, 
                            exclude_problematic = TRUE,
                            min_genes = 5,
                            exclude_drug_metabolism = FALSE,
                            exclude_glycan_biosynthesis = FALSE) {
  
  filtered <- list()
  
  for (name in names(pathways)) {
    genes <- pathways[[name]]
    
    # Skip excluded pathways
    if (exclude_problematic && name %in% EXCLUDED_PATHWAYS) {
      next
    }
    
    # Skip pathways with too few genes
    if (length(genes) < min_genes) {
      next
    }
    
    # Optional: skip drug metabolism pathways
    if (exclude_drug_metabolism) {
      if (grepl("drug|xenobiotics", name, ignore.case = TRUE)) {
        next
      }
    }
    
    # Optional: skip glycan biosynthesis
    if (exclude_glycan_biosynthesis) {
      if (grepl("glycan|glycosyl", name, ignore.case = TRUE)) {
        next
      }
    }
    
    filtered[[name]] <- genes
  }
  
  filtered
}

#' Build Metabolic Graph
#'
#' @description
#' Build metabolic network graph from KEGG pathway definitions with scientific filtering.
#' This function initializes the metabolic knowledge base with:
#' - Full KEGG pathways (~77 filtered from 85 pathways) from scMetabolism
#' - OR curated core pathways (17 pathways, 280+ genes)
#' - Rate-limiting enzyme definitions
#' - Metabolic network structure
#' - Metabolic axis definitions
#'
#' @param species Character. Species code (default "hsa" for human).
#' @param gmt_file Character. Path to custom GMT file (optional).
#' @param use_full_kegg Logical. If TRUE (default), load full KEGG pathways.
#' @param filter_problematic Logical. If TRUE (default), removes biologically 
#'   inappropriate pathways (bacterial, essential AA biosynthesis, etc.)
#' @param min_genes Integer. Minimum genes required per pathway (default 5).
#' @param exclude_drug_metabolism Logical. If TRUE, removes drug metabolism pathways.
#' @param exclude_glycan_biosynthesis Logical. If TRUE, removes glycan biosynthesis.
#'
#' @return A list containing:
#' \describe{
#'   \item{pathways}{Named list of pathway gene vectors}
#'   \item{rate_limiting}{Character vector of rate-limiting enzymes}
#'   \item{network}{igraph object of metabolic network}
#'   \item{metabolic_axes}{Named list of metabolic axis definitions}
#' }
#'
#' @details
#' Pathways excluded by default with filter_problematic=TRUE:
#' \itemize{
#'   \item Neomycin, kanamycin and gentamicin biosynthesis (bacterial pathway)
#'   \item Valine, leucine and isoleucine biosynthesis (essential AA)
#'   \item Phenylalanine, tyrosine and tryptophan biosynthesis (essential AA)
#'   \item D-Arginine and D-ornithine metabolism (1 gene only)
#'   \item D-Glutamine and D-glutamate metabolism
#'   \item Caffeine metabolism
#'   \item Biotin metabolism, Lipoic acid metabolism (< 5 genes)
#' }
#'
#' @examples
#' kb <- build_metabolic_graph()  # 77 filtered pathways
#' kb <- build_metabolic_graph(filter_problematic = FALSE)  # All 85
#' kb <- build_metabolic_graph(use_full_kegg = FALSE)  # Core 17 pathways
#'
#' @export
build_metabolic_graph <- function(species = "hsa", 
                                  gmt_file = NULL, 
                                  use_full_kegg = TRUE,
                                  filter_problematic = TRUE,
                                  min_genes = 5,
                                  exclude_drug_metabolism = FALSE,
                                  exclude_glycan_biosynthesis = FALSE) {
  
  # Try to load from custom GMT file
  if (!is.null(gmt_file) && file.exists(gmt_file)) {
    pathways <- parse_gmt_file(gmt_file)
    pathways <- filter_pathways(pathways, 
                                exclude_problematic = filter_problematic,
                                min_genes = min_genes,
                                exclude_drug_metabolism = exclude_drug_metabolism,
                                exclude_glycan_biosynthesis = exclude_glycan_biosynthesis)
    return(finalize_kb(pathways))
  }
  
  # Try to load full KEGG from bundled GMT
  if (use_full_kegg) {
    # Look for GMT file in package data or relative path
    possible_paths <- c(
      system.file("data", "KEGG_metabolism_nc.gmt", package = "MetaTrajectory"),
      file.path(dirname(getwd()), "data", "KEGG_metabolism_nc.gmt"),
      file.path(getwd(), "..", "data", "KEGG_metabolism_nc.gmt"),
      file.path(getwd(), "data", "KEGG_metabolism_nc.gmt")
    )
    
    for (gmt_path in possible_paths) {
      if (file.exists(gmt_path) && nchar(gmt_path) > 0) {
        pathways <- parse_gmt_file(gmt_path)
        pathways <- filter_pathways(pathways, 
                                    exclude_problematic = filter_problematic,
                                    min_genes = min_genes,
                                    exclude_drug_metabolism = exclude_drug_metabolism,
                                    exclude_glycan_biosynthesis = exclude_glycan_biosynthesis)
        return(finalize_kb(pathways))
      }
    }
    
    warning("KEGG GMT file not found. Falling back to core pathways.")
  }
  
  # Fall back to core pathways
  build_core_pathways()
}

#' Finalize Knowledge Base
#'
#' @description
#' Add rate-limiting enzymes, axes, and network to pathways.
#'
#' @param pathways Named list of pathway genes.
#'
#' @return Complete knowledge base list.
#'
#' @keywords internal
finalize_kb <- function(pathways) {
  rate_limiting <- get_rate_limiting_enzymes()
  metabolic_axes <- get_metabolic_axes()
  network <- build_pathway_network(names(pathways))
  
  list(
    pathways = pathways,
    rate_limiting = rate_limiting,
    network = network,
    metabolic_axes = metabolic_axes
  )
}

#' Get Rate-Limiting Enzymes
#'
#' @description
#' Get list of rate-limiting enzymes.
#'
#' @return Character vector.
#'
#' @keywords internal
get_rate_limiting_enzymes <- function() {
  c(
    # Glycolysis
    "HK1", "HK2", "HK3", "GCK",
    "PFKM", "PFKL", "PFKP",
    "PKM", "PKLR",
    # Gluconeogenesis
    "PC", "PCK1", "PCK2",
    "FBP1", "FBP2",
    "G6PC",
    # TCA cycle
    "CS",
    "IDH1", "IDH2", "IDH3A",
    "OGDH",
    # OXPHOS
    "NDUFS1", "SDHA", "UQCRC1", "COX4I1", "ATP5F1A",
    # Fatty acid
    "ACACA", "ACACB", "FASN",
    "CPT1A", "CPT1B",
    # Cholesterol
    "HMGCR",
    # Amino acid
    "GLS", "GLS2", "GLUL",
    "PHGDH",
    # Nucleotide
    "RRM1", "RRM2",
    "IMPDH1", "IMPDH2",
    # Hypoxia
    "LDHA", "PDK1"
  )
}

#' Get Metabolic Axes
#'
#' @description
#' Get metabolic axis definitions.
#'
#' @return Named list of axes.
#'
#' @keywords internal
get_metabolic_axes <- function() {
  list(
    Glycolysis_vs_OXPHOS = list(
      positive = c("HK1", "HK2", "PFKM", "PFKL", "PKM", "LDHA", "PDK1"),
      negative = c("CS", "IDH2", "SDHA", "ATP5F1A", "COX4I1", "NDUFS1"),
      description = "Warburg effect: glycolytic vs oxidative metabolism"
    ),
    Anabolism_vs_Catabolism = list(
      positive = c("ACACA", "FASN", "ACLY", "PHGDH", "PSAT1"),
      negative = c("CPT1A", "GLS", "ACADM", "HADHA"),
      description = "Biosynthesis vs breakdown"
    ),
    Proliferative_vs_Quiescent = list(
      positive = c("PKM", "LDHA", "GLS", "PHGDH", "RRM1", "RRM2", "TYMS"),
      negative = c("IDH2", "CS", "CPT1A", "ACADM"),
      description = "High vs low proliferation metabolic profile"
    ),
    Redox_Stress = list(
      positive = c("G6PD", "PGD", "GSR", "GCLC", "NQO1"),
      negative = c("SOD1", "SOD2", "CAT"),
      description = "Antioxidant response activation"
    ),
    Lipid_Synthesis_vs_Oxidation = list(
      positive = c("ACACA", "FASN", "SCD", "ACLY"),
      negative = c("CPT1A", "CPT1B", "ACADM", "HADHA"),
      description = "Lipogenesis vs lipolysis"
    )
  )
}

#' Build Core Pathways
#'
#' @description
#' Build knowledge base with core 17 pathways.
#'
#' @return Knowledge base list.
#'
#' @keywords internal
build_core_pathways <- function() {
  
  # Define core metabolic pathways (17 pathways, matching Python implementation)
  pathways <- list(
    # Carbohydrate metabolism
    Glycolysis = c(
      "HK1", "HK2", "HK3", "GCK",
      "GPI",
      "PFKM", "PFKL", "PFKP",
      "ALDOA", "ALDOB", "ALDOC",
      "TPI1",
      "GAPDH",
      "PGK1", "PGK2",
      "PGAM1", "PGAM2",
      "ENO1", "ENO2", "ENO3",
      "PKM", "PKLR"
    ),
    
    Gluconeogenesis = c(
      "PC",
      "PCK1", "PCK2",
      "FBP1", "FBP2",
      "G6PC", "G6PC2", "G6PC3"
    ),
    
    TCA_Cycle = c(
      "CS",
      "ACO1", "ACO2",
      "IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G",
      "OGDH", "OGDHL",
      "SUCLA2", "SUCLG1", "SUCLG2",
      "SDHA", "SDHB", "SDHC", "SDHD",
      "FH",
      "MDH1", "MDH2"
    ),
    
    Pentose_Phosphate = c(
      "G6PD",
      "PGLS",
      "PGD",
      "RPE",
      "RPIA",
      "TKT",
      "TALDO1"
    ),
    
    # Oxidative phosphorylation (83 genes)
    OXPHOS = c(
      # Complex I (NADH dehydrogenase)
      "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS4", "NDUFS5", "NDUFS6", "NDUFS7", "NDUFS8",
      "NDUFV1", "NDUFV2", "NDUFV3",
      "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5", "NDUFA6", "NDUFA7", "NDUFA8", 
      "NDUFA9", "NDUFA10", "NDUFA11", "NDUFA12", "NDUFA13",
      "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5", "NDUFB6", "NDUFB7", "NDUFB8", 
      "NDUFB9", "NDUFB10", "NDUFB11",
      "NDUFC1", "NDUFC2",
      # Complex II (Succinate dehydrogenase)
      "SDHA", "SDHB", "SDHC", "SDHD",
      # Complex III (Cytochrome bc1)
      "UQCRC1", "UQCRC2", "UQCRFS1", "UQCRB", "UQCRH", "UQCRQ", "UQCR10", "UQCR11",
      "CYC1",
      # Complex IV (Cytochrome c oxidase)
      "COX4I1", "COX4I2", "COX5A", "COX5B", "COX6A1", "COX6A2", "COX6B1", "COX6B2", "COX6C",
      "COX7A1", "COX7A2", "COX7B", "COX7C", "COX8A", "COX8C",
      "MT-CO1", "MT-CO2", "MT-CO3",
      # Complex V (ATP synthase)
      "ATP5F1A", "ATP5F1B", "ATP5F1C", "ATP5F1D", "ATP5F1E",
      "ATP5PB", "ATP5MC1", "ATP5MC2", "ATP5MC3", "ATP5PD", "ATP5ME", "ATP5PF", "ATP5MG", 
      "ATP5MPL", "ATP5IF1"
    ),
    
    # Lipid metabolism
    Fatty_Acid_Synthesis = c(
      "ACACA", "ACACB",
      "FASN",
      "ACLY",
      "SCD",
      "ELOVL1", "ELOVL2", "ELOVL3", "ELOVL4", "ELOVL5", "ELOVL6", "ELOVL7",
      "FADS1", "FADS2"
    ),
    
    Fatty_Acid_Oxidation = c(
      "CPT1A", "CPT1B", "CPT1C",
      "CPT2",
      "SLC25A20",
      "ACADVL", "ACADL", "ACADM", "ACADS",
      "HADHA", "HADHB",
      "ECHS1",
      "HADH",
      "ACAA1", "ACAA2"
    ),
    
    Cholesterol_Synthesis = c(
      "HMGCS1", "HMGCS2",
      "HMGCR",
      "MVK",
      "PMVK",
      "MVD",
      "IDI1", "IDI2",
      "FDPS",
      "FDFT1",
      "SQLE",
      "LSS",
      "DHCR7", "DHCR24"
    ),
    
    # Amino acid metabolism
    Glutamine_Metabolism = c(
      "GLS", "GLS2",
      "GLUL",
      "GLUD1", "GLUD2",
      "GOT1", "GOT2",
      "GPT", "GPT2",
      "SLC1A5", "SLC38A1", "SLC38A2"
    ),
    
    Serine_Glycine_Metabolism = c(
      "PHGDH",
      "PSAT1",
      "PSPH",
      "SHMT1", "SHMT2",
      "MTHFD1", "MTHFD2", "MTHFD1L", "MTHFD2L"
    ),
    
    Branched_Chain_AA = c(
      "BCAT1", "BCAT2",
      "BCKDHA", "BCKDHB",
      "DBT",
      "DLD"
    ),
    
    One_Carbon_Metabolism = c(
      "MTHFR",
      "MTR",
      "MTRR",
      "MAT1A", "MAT2A", "MAT2B",
      "AHCY",
      "CBS",
      "CTH"
    ),
    
    # Nucleotide metabolism
    Purine_Synthesis = c(
      "PPAT",
      "GART",
      "PFAS",
      "PAICS",
      "ADSL",
      "ATIC",
      "IMPDH1", "IMPDH2",
      "GMPS"
    ),
    
    Pyrimidine_Synthesis = c(
      "CAD",
      "DHODH",
      "UMPS",
      "UCK1", "UCK2",
      "CMPK1", "CMPK2",
      "NME1", "NME2",
      "RRM1", "RRM2",
      "TYMS"
    ),
    
    # Hypoxia and lactate
    Hypoxia_Metabolism = c(
      "LDHA", "LDHB", "LDHC",
      "SLC16A1", "SLC16A3",
      "PDK1", "PDK2", "PDK3", "PDK4",
      "PDHA1", "PDHB",
      "HIF1A",
      "EPAS1",
      "VEGFA",
      "SLC2A1", "SLC2A3"
    ),
    
    # Ketone body metabolism
    Ketogenesis = c(
      "HMGCS2",
      "HMGCL",
      "BDH1", "BDH2",
      "OXCT1", "OXCT2"
    ),
    
    # Reactive oxygen species
    Antioxidant = c(
      "SOD1", "SOD2", "SOD3",
      "CAT",
      "GPX1", "GPX2", "GPX3", "GPX4",
      "GSR",
      "GCLC", "GCLM",
      "GSS",
      "TXN", "TXN2",
      "TXNRD1", "TXNRD2",
      "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6",
      "NQO1"
    )
  )
  
  # Use finalize_kb to add rate_limiting, axes, and network
  finalize_kb(pathways)
}

#' Build Pathway Network
#'
#' @description
#' Build igraph network representing metabolic pathway connections.
#'
#' @param pathway_names Character vector of pathway names.
#'
#' @return An igraph object.
#'
#' @keywords internal
build_pathway_network <- function(pathway_names) {
  # Define connections (matching Python implementation)
  edges <- matrix(c(
    # Central carbon metabolism
    "Glycolysis", "TCA_Cycle",
    "Glycolysis", "Pentose_Phosphate",
    "Glycolysis", "Hypoxia_Metabolism",
    "Gluconeogenesis", "Glycolysis",
    "TCA_Cycle", "OXPHOS",
    "TCA_Cycle", "Gluconeogenesis",
    # Lipid metabolism
    "Glycolysis", "Fatty_Acid_Synthesis",
    "TCA_Cycle", "Fatty_Acid_Synthesis",
    "Fatty_Acid_Oxidation", "TCA_Cycle",
    "Fatty_Acid_Oxidation", "Ketogenesis",
    "Cholesterol_Synthesis", "Fatty_Acid_Synthesis",
    # Amino acid metabolism
    "TCA_Cycle", "Glutamine_Metabolism",
    "Glutamine_Metabolism", "TCA_Cycle",
    "Glycolysis", "Serine_Glycine_Metabolism",
    "Serine_Glycine_Metabolism", "One_Carbon_Metabolism",
    "Branched_Chain_AA", "TCA_Cycle",
    # Nucleotide metabolism
    "Pentose_Phosphate", "Purine_Synthesis",
    "Pentose_Phosphate", "Pyrimidine_Synthesis",
    "One_Carbon_Metabolism", "Purine_Synthesis",
    "One_Carbon_Metabolism", "Pyrimidine_Synthesis",
    "Glutamine_Metabolism", "Purine_Synthesis",
    "Glutamine_Metabolism", "Pyrimidine_Synthesis",
    # Redox and stress
    "Pentose_Phosphate", "Antioxidant",
    "OXPHOS", "Antioxidant",
    # Hypoxia connections
    "Hypoxia_Metabolism", "Glycolysis"
  ), ncol = 2, byrow = TRUE)
  
  # Filter to available pathways
  valid_edges <- edges[edges[, 1] %in% pathway_names & edges[, 2] %in% pathway_names, , drop = FALSE]
  
  if (nrow(valid_edges) == 0) {
    return(igraph::make_empty_graph(n = length(pathway_names), directed = TRUE))
  }
  
  g <- igraph::graph_from_edgelist(valid_edges, directed = TRUE)
  
  # Add missing nodes
  missing <- setdiff(pathway_names, igraph::V(g)$name)
  if (length(missing) > 0) {
    g <- igraph::add_vertices(g, length(missing), name = missing)
  }
  
  g
}

#' Get All Metabolic Genes
#'
#' @description
#' Get all unique genes from all pathways.
#'
#' @param kb Metabolic knowledge base from build_metabolic_graph().
#'
#' @return Character vector of gene symbols.
#'
#' @export
get_all_metabolic_genes <- function(kb) {
  unique(unlist(kb$pathways))
}

#' Get Pathway for Gene
#'
#' @description
#' Find which pathways contain a specific gene.
#'
#' @param kb Metabolic knowledge base.
#' @param gene Character. Gene symbol.
#'
#' @return Character vector of pathway names.
#'
#' @export
get_pathway_for_gene <- function(kb, gene) {
  pathways <- names(kb$pathways)[sapply(kb$pathways, function(x) gene %in% x)]
  pathways
}

#' Compute Pathway Distance
#'
#' @description
#' Compute distance between two pathways in the metabolic network.
#'
#' @param kb Metabolic knowledge base.
#' @param pathway1 First pathway name.
#' @param pathway2 Second pathway name.
#'
#' @return Numeric distance.
#'
#' @export
compute_pathway_distance <- function(kb, pathway1, pathway2) {
  if (is.null(kb$network)) {
    return(NA)
  }
  
  g <- igraph::as_undirected(kb$network)
  
  tryCatch({
    path <- igraph::shortest_paths(g, from = pathway1, to = pathway2)
    length(path$vpath[[1]]) - 1
  }, error = function(e) {
    Inf
  })
}

#' Knowledge Base Summary
#'
#' @description
#' Print summary of the metabolic knowledge base.
#'
#' @param kb Metabolic knowledge base.
#'
#' @return Invisibly returns the kb.
#'
#' @export
kb_summary <- function(kb) {
  cat("MetabolicKnowledgeBase Summary:\n")
  cat(sprintf("  - Pathways: %d\n", length(kb$pathways)))
  cat(sprintf("  - Total genes: %d\n", length(get_all_metabolic_genes(kb))))
  cat(sprintf("  - Rate-limiting enzymes: %d\n", length(kb$rate_limiting)))
  cat(sprintf("  - Metabolic axes: %d\n", length(kb$metabolic_axes)))
  cat(sprintf("  - Network edges: %d\n", igraph::ecount(kb$network)))
  invisible(kb)
}
