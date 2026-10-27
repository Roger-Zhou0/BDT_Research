# Dependencies, attribution, and release rights

This archive contains frozen project implementations and new packaging helpers.
The directory name `native` means the original project implementation, not a
redistribution of a third-party framework. PyTorch, NumPy, SciPy and Matplotlib
are imported from separately installed distributions; their source, binaries,
font files and license texts are not bundled here. Their existing terms remain
applicable to those separate distributions.

The model and update code uses standard Transformer, SwiGLU and AdamW ideas.
The GS-derived window follows the Gradient Smoothing method attributed in the
paper to Meng, Sugolov and Papyan (2026). It is a project implementation of the
window equation, not a claim to distribute or execute an unchanged upstream
Gradient Smoothing repository. BCT1 combines this carrier with budget exchange
and attention BDT; it is not standalone GS.

This packaging pass does not select a new repository-wide open-source license
or assert ownership of unexamined upstream code. Before public redistribution,
the authors should verify provenance and any applicable notices for every
component and choose the intended release license. Review-time inspection and
public open-source redistribution are distinct release decisions. No legal
notice should be deleted merely to improve anonymity.
