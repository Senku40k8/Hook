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

## Utilisation

Aucun réglage préalable n'est nécessaire : lancez le bot, puis donnez le focus à la fenêtre
du jeu dans les 3 secondes.

```bash
python hook.py                       # boucle infinie
python hook.py --cycles 10 -v        # 10 cycles, logs détaillés
python hook.py --invert              # si Q/D agissent dans le sens opposé
python hook.py --input pyautogui     # touches via PyAutoGUI au lieu des scancodes
python hook.py --reset               # oublie les zones mémorisées
```

Au premier cycle, la bannière et la jauge sont cherchées sur tout l'écran. Pour la jauge, une
barre verte n'est retenue que si la ligne jaune est à côté, ce qui évite de la confondre avec
le vert du décor. Dès qu'un élément est trouvé, sa zone est verrouillée et mémorisée dans
`hook_config.json` (propre à votre écran, ignoré par git). Les cycles suivants, et les
prochains lancements, ne capturent plus que ces zones, ce qui est plus rapide.

Le template de la bannière (`templates/fish_on_hook.png`) contient seulement le texte, sans
le décor. Si l'interface du jeu change, refaites une capture recadrée sur le texte uniquement.

## Si la jauge n'est pas détectée (facultatif)

Les couleurs par défaut de la zone verte et de la ligne jaune sont génériques. Si le bot
écrit « Jauge non détectée », mesurez les vraies couleurs avec la **jauge visible à l'écran** :

```bash
python hook.py --setup
```

Après 10 secondes, une capture d'écran s'ouvre. Tracez successivement à la souris (Entrée
pour valider, C pour ignorer une étape) :

1. la zone où apparaît la bannière ;
2. la zone de la jauge, un peu plus large que la barre ;
3. un rectangle à l'intérieur de la zone verte ;
4. un rectangle serré autour de la ligne jaune.

Pour vérifier, gardez la jauge à l'écran et lancez `python hook.py --calibrate`. Le dossier
`debug/` reçoit alors la capture, les masques vert et jaune, ainsi qu'une image annotée
(rectangle magenta = zone verte, trait rouge = indicateur).

### Envoi des touches

Sous Windows, les touches sont envoyées par défaut comme un vrai clavier (`SendInput` +
scancodes), car beaucoup de jeux ignorent les touches virtuelles de PyAutoGUI. Le scancode
est calculé selon la disposition clavier active : `q` correspond bien à la touche marquée Q,
en AZERTY comme en QWERTY. Les options en ligne de commande (`--gauge-region`,
`--banner-region`, `--invert`, `--input`) priment sur `hook_config.json`.

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
