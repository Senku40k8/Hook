# Hook

Boucle de test d'interface automatisée en Python (PyAutoGUI + OpenCV) :

1. appui sur `F`, puis latence aléatoire de 0.5 s à 1.5 s ;
2. attente de la bannière « Fish on the hook! Press the button to reel it! », puis nouvel appui sur `F` ;
3. suivi de la jauge (zone verte + indicateur jaune) avec des pressions courtes sur `Q` (gauche) et `D` (droite), jusqu'à ce que la jauge disparaisse ;
4. pause aléatoire de 3.0 s à 4.5 s, puis retour à l'étape 1.

## Installation

```bash
pip install -r requirements.txt
```

## Calibration (à faire une fois)

1. **Template de la bannière** : quand la bannière est affichée, faites une capture et
   recadrez uniquement le texte. Enregistrez l'image dans `templates/fish_on_hook.png`.
2. **Zones et couleurs** : avec la jauge visible à l'écran, lancez :
   ```bash
   python hook.py --calibrate --gauge-region L T W H
   ```
   Le dossier `debug/` reçoit alors la capture, les masques vert et jaune, et une image
   annotée (rectangle magenta = zone verte, trait rouge = indicateur). Si les masques ne
   correspondent pas, ajustez `green_lower/upper` et `yellow_lower/upper` dans `Config`
   (format HSV d'OpenCV : H de 0 à 179).

Indiquer `--gauge-region` et `--banner-region` est recommandé : la capture est plus rapide
et il y a moins de faux positifs.

## Utilisation

```bash
python hook.py                       # boucle infinie
python hook.py --cycles 10 -v        # 10 cycles, logs détaillés
python hook.py --invert              # si Q/D agissent dans le sens opposé
```

Vous avez 3 secondes après le lancement pour donner le focus à la fenêtre cible.

## Arrêt

- touche **Échap** (vérifiée en continu, même pendant les pauses) ;
- **fail-safe** PyAutoGUI : amener la souris dans un coin de l'écran ;
- **Ctrl+C** dans le terminal.

Les touches sont toujours relâchées à l'arrêt.

## Tests

```bash
python -m unittest discover -s tests
```

Ces tests vérifient les détecteurs et le correcteur sur des images synthétiques. Ils ne
nécessitent ni écran ni fenêtre cible.
