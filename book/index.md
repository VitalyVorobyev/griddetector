# GridDetector

**GridDetector** is an edge-based grid/chessboard detector written in Rust. It builds an image pyramid, extracts line segments (LSD-like), groups them into two dominant line families to estimate vanishing points, composes a coarse homography, filters outliers, refines across the pyramid, and indexes the grid.  
Target use cases: fast CPU detection in live streams, with robust diagnostics.  
License: MIT.
