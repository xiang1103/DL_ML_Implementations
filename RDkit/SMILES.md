# SMILES Strings 
- Smiles(Simplified Molecular Input Line Entry System) strings are 2D representations of any molecules created in 1980s 
-  Because of its 2D structure in a **sequence**, it's good for machine processing. Think of transformers being good at processing sequential text. The sequential encoding of molecules now allow machines to process and understand molecules as sequence of letters  
  
# Encoding Rules 
*How to encode molecules?*  
- SMILES follow specific encoding rules. That means there are basic **syntatical rules** to make a SMILES string valid. To make a distinction, it's important to note that there is *no specific* way we must follow to encode a molecule, but there are basic guidelines to generate valid strings. 
  ## Syntatic Rules 
  - **Atoms and Bonds**: All atoms are encoded by their representation on the Periodic table (O for Oxygen). The bonds are represented by specific symbols. Such as {=} for double bond. Single bonds are not encoded  [Check the here](https://archive.epa.gov/med/med_archive_03/web/html/smiles.html)
  - **Simple Bonds**: Hydrogen atoms are omitted by default (implicit). But we can specify the hydrogen bonding explicitly as well. 
    - There are consequences of omitting Hydrogen bonding. Without explicit Hydrogen atoms, programs add Hydrogen to "fill in the blank". 