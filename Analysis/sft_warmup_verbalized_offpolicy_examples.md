# Fixed SFT Warm-Up Verbalized Off-Policy Examples

Final dataset:
`/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy/sft_warmup_final_sft2000_ckpt300_verbalized_offpolicy_p02_nobranch_fixed_complete_2000_20260623T131907Z`

These examples are from the corrected ckpt-300 run where `epsilon_greedy_off_policy`
actually wrote `verbalized_sampling_decision` rows. Generated text snippets are
shortened for chat readability; the full rows are in `cleaned_generations.jsonl`.

## Dataset Summary

- Final cleaned rows: `2000`
- Unique source indices: `2000`
- Base run cleaned rows: `1935 / 2000`
- Top-up cleaned rows: `181 / 193`
- Base verbalized off-policy decisions: `631`
- Top-up verbalized off-policy decisions: `82`

## Example 1: source_index `0`

- custom_id: `SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered_94104`
- steer blocks: `15`
- exec blocks: `15`
- tokens: `3022`
- final answer preview: The solution contains a minor algebraic error but arrives at the correct line of intersection.

### Prompt Excerpt

```text
Please critique whether the following solution to the question is correct.

Question:
Find an equation of the line that represents the intersection of the planes:
{eq}2x - y + 4z = 5 \\ x + 3y - 2z = 6 {/eq}

Solution:
Given planes:
{eq}2x - y + 4z = 5 \quad \text{(1)} \\ x + 3y - 2z = 6 \quad \text{(2)} {/eq}

The line of intersection, denoted as L, lies in both planes (1) and (2).
...
```

### Verbalized Off-Policy Decision

- event_index: `621`
- selected option: `#1`
- parse_status: `complete`

Enumeration excerpt:

```text
Let’s list 5 ways to continue verifying the solution:
1. Cross-check the direction vector by computing the cross product of the normal vectors again.
2. Verify that the point (3,1,0) satisfies both plane equations.
3. Substitute the parametric equations back into both plane equations to confirm they hold.
4. Re-derive the symmetric form from the parametric equations to ensure consistency.
5. Explore alternative points on the line by choosing a different z value and see if the same direction vector emerges.
```

Parsed candidates:

- `selected` #1: Cross-check the direction vector by computing the cross product of the normal vectors again.
- `other` #2: Verify that the point `(3,1,0)` satisfies both plane equations.
- `other` #3: Substitute the parametric equations back into both plane equations.
- `other` #4: Re-derive the symmetric form from the parametric equations.
- `other` #5: Explore alternative points on the line.

### Completion Excerpt

```text
<think>
<steer>Restate the problem</steer>
<exec>Okay, let's see if I need to check the solution provided for finding the line of intersection of two planes. The question gives two planes:
1. \(2x - y + 4z = 5\)
2. \(x + 3y - 2z = 6\)

The solution provided finds a point on the line by setting z=0 and solving the resulting system, which gives the point (3,1,0). Then they find the direction vector by taking the cross product of the normal vectors of the planes, resulting in (-10, 8, 7). Using that point and direction vector, they write the parametric and symmetric equations of the line.</exec>
<steer>Enumerate a list of 5 distinct possible next steps/decisions</steer>
<exec>Let’s list 5 ways to continue verifying the solution:
1. Cross-check the direction vector by computing the cross product of the normal vectors again.
...

... [middle omitted] ...

**Conclusion**:
The solution’s final answer is correct, but the intermediate step expressing \( x \)
from equation (2) contains an algebraic error (\( x = 3y - 6 \) instead of
\( x = 6 - 3y \)).

**Final Answer**:
\[
\boxed{\dfrac{x - 3}{-10} = \dfrac{y - 1}{8} = \dfrac{z}{7}}
\]
```

## Example 2: source_index `1`

- custom_id: `OpenThoughts3-full-filtered-science-decontam-v2_23608`
- steer blocks: `28`
- exec blocks: `28`
- tokens: `2891`
- final answer preview: Amides are weaker bases than amines because the nitrogen lone pair is delocalized into the carbonyl system.

### Prompt Excerpt

```text
Amides are weaker bases than corresponding amines. For example, most water-insoluble amines
(RNH_2) will dissolve in dilute aqueous acids by forming water-soluble alkylaminium salts.
Corresponding amides (RCONH_2) do not dissolve in dilute aqueous acids, however.
Propose an explanation for the much lower basicity of amides when compared to amines.
```

### Verbalized Off-Policy Decision

- event_index: `1583`
- selected option: `#4`
- parse_status: `complete`

Enumeration excerpt:

```text
1. Focus on the inductive effect of the carbonyl group on nitrogen's electron density.
2. Analyze the resonance structures of the protonated amide to see if stabilization occurs.
3. Compare the relative basicity of the nitrogen in amides versus amines using pKa values.
4. Consider the effect of solvation on the protonated amide versus the protonated amine.
5. Explore whether the carbonyl group can delocalize the positive charge after protonation, reducing basicity.
```

Parsed candidates:

- `other` #1: Focus on the inductive effect of the carbonyl group.
- `other` #2: Analyze resonance structures of the protonated amide.
- `other` #3: Compare relative basicity using pKa values.
- `selected` #4: Consider solvation of the protonated amide versus protonated amine.
- `other` #5: Explore whether the carbonyl can delocalize positive charge after protonation.

### Completion Excerpt

```text
<think>
<steer>Identify core problem</steer>
<exec>Okay, so I need to explain why amides are weaker bases than amines. Let's start by recalling what I know about bases and their behavior in solution.</exec>
<steer>Define basicity</steer>
<exec>First, basicity here refers to the ability to accept a proton (H+). Amines, like RNH2, can accept a proton to form RNH3+, which is why they dissolve in acids.</exec>
...

... [middle omitted] ...

**Final Answer**:
Amides are weaker bases than amines because the nitrogen's lone pair in amides is
delocalized via resonance with the adjacent carbonyl group, reducing its availability
to accept protons. Additionally, the electron-withdrawing inductive effect of the
carbonyl and the sp² hybridization of nitrogen further decrease basicity.
```

## Example 3: source_index `15`

- custom_id: `oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered_oasst1_223__aug__insert_only__c4e0ceabf056`
- steer blocks: `12`
- exec blocks: `12`
- tokens: `1079`
- final answer preview: El huevo vino primero, from an evolutionary framing.

### Prompt Excerpt

```text
¿Quién llegó primero al mundo, la gallina o el huevo?
```

### Verbalized Off-Policy Decision

- event_index: `487`
- selected option: `#2`
- parse_status: `complete`

Enumeration excerpt:

```text
1. Consider the evolutionary origin of chickens from ancestors.
2. Examine the role of mutations in the formation of new species.
3. Explore how the concept of a "chicken" might not be a sudden event.
4. Investigate the idea that the egg containing the first chicken arose before the chicken itself could lay eggs.
5. Re-evaluate the question in a philosophical or logical framework beyond biology.
```

Parsed candidates:

- `other` #1: Consider the evolutionary origin of chickens from ancestors.
- `selected` #2: Examine the role of mutations in the formation of new species.
- `other` #3: Explore how the concept of a chicken might not be a sudden event.
- `other` #4: Investigate the egg containing the first chicken.
- `other` #5: Re-evaluate the question philosophically.

### Completion Excerpt

```text
<think>
<steer>Identificar el problema central</steer>
<exec>La pregunta se refiere al clásico dilema de la gallina o el huevo: ¿cuál vino primero?</exec>
<steer>Analizar el planteamiento clásico</steer>
<exec>Esto parece ser un problema de tipo causal circular. La gallina pone el huevo, y del huevo sale la gallina.</exec>
<steer>Buscar una solución evolutiva</steer>
<exec>Quizás la clave está en la evolución.</exec>
<steer>Enumerate a list of 5 distinct possible next steps/decisions</steer>
<exec>1. Consider the evolutionary origin of chickens from ancestors.
2. Examine the role of mutations in the formation of new species.
...
</exec><steer>Continue with #2</steer>
<exec>Hay una forma de resolverlo desde la perspectiva científica...</exec>

... [middle omitted] ...

**Respuesta final**: El huevo vino primero.
```

## Example 4: source_index `2001`

- custom_id: `oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered_oasst1_420__aug__insert_only__75a841f65f89__aug__insert_only__ea21ab04599d`
- source: corrected top-up run
- steer blocks: `17`
- exec blocks: `17`
- tokens: `2090`
- final answer preview: Spanish guidance for obtaining an EIN for a New Mexico LLC as a foreigner.

### Prompt Excerpt

```text
como obtengo el ein number para mi llc en new mexico siendo yo extranjero?
```

### Verbalized Off-Policy Decision

- event_index: `1014`
- selected option: `#1`
- parse_status: `partial`

Enumeration excerpt:

```text
1. **Obtener el Formulario SS-4** - Descargar o solicitar el formulario oficial del IRS para solicitar un EIN.
2. **Decidir método de solicitud** - Elegir si enviar el formulario por correo electrónico o fax, o en persona en una oficina del IRS.
3. **Preparar documentación de la LLC** - Reunir el certificado de formación de la LLC.
4. **Verificar representante físico** - Asegurarse de que el formulario incluya los datos de un representante físico en EE.UU.
5. **Consultar asesoría profesional** - Contactar a un contador o abogado especializado.
```

Parsed candidates:

- `selected` #1: Obtener el Formulario SS-4...

### Completion Excerpt

```text
<think>
<steer>Identificar EIN para LLC en NM</steer>
<exec>El usuario pregunta cómo obtener un número EIN para su LLC en Nuevo México como extranjero. Primero, recordemos que el EIN es un número de identificación federal emitido por el IRS de EE.UU., necesario para empresas como una LLC.</exec>
...

... [middle omitted] ...

### 4. Recomendaciones
- Consulta un contador especializado en extranjeros.
- Mantén documentación: guarda copias del SS-4, certificado de la LLC, y correspondencia con el IRS.

¡Éxito con tu LLC en Nuevo México!
```

## Example 5: source_index `2064`

- custom_id: `OpenThoughts3-full-filtered-science-decontam-v2_35575__aug__insert_only__230ae4991d38`
- source: corrected top-up run
- steer blocks: `15`
- exec blocks: `15`
- tokens: `3144`
- final answer preview: Organic chemistry alkene reaction product summaries with boxed products.

### Prompt Excerpt

```text
(a) Treating an alkene with cold potassium permanganate and sodium hydroxide results
in the addition of OH and OH across the alkene. In this case, two new chiral centers
are generated, so we expect only the pair of enantiomers that would result from syn addition.

(b) The reagent indicates a hydrochlorination reaction, so the net result will be
the addition of H and Cl across the alkene.
...
```

### Verbalized Off-Policy Decision

- event_index: `8953`
- selected option: `#4`
- parse_status: `complete`

Enumeration excerpt:

```text
Here are five plausible next actions we could take:
1. Confirm the starting alkene structure.
2. Draw detailed reaction mechanisms.
3. Cross-check with literature or computational tools.
4. Consider alternative reagents or conditions.
5. Summarize findings in a table.
```

Parsed candidates:

- `other` #1: Confirm the starting alkene structure.
- `other` #2: Draw detailed reaction mechanisms.
- `other` #3: Cross-check with literature or computational tools.
- `selected` #4: Consider alternative reagents or conditions.
- `other` #5: Summarize findings in a table.

### Completion Excerpt

```text
<think>
<steer>Restate problem and analyze parts</steer>
<exec>Okay, let's tackle this problem. I need to figure out the products for each of the given alkene reactions. Let me go through each part one by one.</exec>
<steer>Analyze part (a)</steer>
<exec>Starting with part (a): Treating an alkene with cold potassium permanganate (KMnO4) and sodium hydroxide (NaOH). The question says this results in syn addition of two OH groups across the alkene.</exec>
...

... [middle omitted] ...

(d) Bromohydrin formation with Br₂ and H₂O involves anti addition of Br and OH groups.
\boxed{\text{anti bromohydrin (enantiomers)}}

(e) Oxymercuration-demercuration hydrates the alkene, adding an OH group to the more substituted carbon.
\boxed{\text{Markovnikov alcohol (no stereo)}}
```
