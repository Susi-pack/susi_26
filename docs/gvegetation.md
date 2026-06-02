Copy-pasted from gvegetation.py directly:

SOURCE:
    Muukkonen & Makipaa, 2006. Bor.Env.Res. 11, 355-369.\n
AUTHOR:
    Samuli Launiainen 18.06.2014, Modified for array operations by Ari Laurén 13.4.2020 \n
NOTE:
     Multi-regression models not yet tested!
     In model equations independent variables named differently to M&M (2006): here x[0] = z1, x[1]=z2, ... x[7]=z8 and x[8]=z10\n
     \n

     Site nutrient fertility class (sfc) at mires:\n
         1: herb-rich hw-spruce swamps, pine mires, fens,
         2: V.myrtillus / tall sedge spruce swamps, tall sedge pine fens, tall sedge fens,
         3: Carex clobularis / V.vitis-idaea swamps, Carex globularis pine swamps, low sedge (oligotrophic) fens,
         4: Low sedge, dwarf-shrub & cottongrass pine bogs, ombo-oligotrophic bogs,
         5: S.fuscum pine bogs, ombotrophic and S.fuscum low sedge bogs.
     Drainage status x[8] at mires (Paavilainen & Paivanen, 1995):
         1: undrained
         2: Recently draines, slight effect on understory veg., no effect on stand
         3: Transforming drained mires, clear effect on understory veg and stand
         4: Transforming drained mires, veget. resembles upland forest site type, tree-stand forest-like.


# General Parameters

Computed:
   - total biomass and bottom layer; field layer is gained as a difference of tot and bottom layer (more cohrent results)
   - N and P storage in the each pixel
   - annual use of N and P due to litterfall
Muukkonen Mäkipää 2005 upland sites: field layer contains dwarf shrubs and (herbs + grasses), see Fig 1
    share     dwarf shrubs     herbs 
    - Pine       91%            9%
    - Spruce     71%            29%
    - broad l    38%            62%
Peatland sites (assumption):
    share      dwarf shrubs    herbs
    - Pine bogs    90%          10%
    - Spruce mires 50%          50%
Palviainen et al. 2005 Ecol Res (2005) 20: 652–660, Table 2
Nutrient concentrations for
                        N              P           K
    - Dwarf shrubs      1.2%         1.0 mg/g     4.7 mg/g
    - herbs & grasses   1.8%         2.0 mg/g    15.1 mg/g
    - upland mosses     1.25%        1.4 mg/g     4.3 mg/g
Nutrient concentrations for sphagna (FIND):
                        N              P     for N :(Bragazza et al Global Change Biology (2005) 11, 106–114, doi: 10.1111/j.1365-2486.2004.00886.x)
    - sphagnum          0.6%           1.4 mg/g     (Palviainen et al 2005)   
Annual litterfall proportions from above-ground biomass (Mälkönen 1974, Tamm 1953):
    - Dwarf shrubs          0.2
    - herbs & grasses        1
    - mosses                0.3
    Tamm, C.O. 1953. Growth, yield and nutrition in carpets of a forest moss (Hylocomium splendens). Meddelanden från Statens Skogsforsknings Institute 43 (1): 1-140.

We assume retranslocation of N and P away from senescing tissues before litterfall:
                        N           P
    - Dwarf shrubs     0.5         0.5
    - Herbs & grasses  0.5         0.5
    - mossess          0.0         0.0

Turnover of total biomass including the belowground biomass is assumed to be 1.2 x above-ground biomass turnover
