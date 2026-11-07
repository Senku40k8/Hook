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

## Réglage (à faire une fois)

Le template de la bannière est fourni : `templates/fish_on_hook.png` contient seulement le
texte, sans le décor, pour que la détection ne dépende pas de l'arrière-plan. Si l'interface
du jeu change, refaites une capture recadrée sur le texte uniquement.

Avec la **jauge visible à l'écran**, lancez :

```bash
python hook.py --setup
```

Après 3 secondes, une capture d'écran s'ouvre. Tracez successivement à la souris (Entrée
pour valider, C pour ignorer une étape) :

1. la zone où apparaît la bannière (facultatif, accélère la détection) ;
2. la zone de la jauge, un peu plus large que la barre (**obligatoire** : sans elle, le
   vert du décor peut être pris pour la jauge) ;
3. un rectangle à l'intérieur de la zone verte ;
4. un rectangle serré autour de la ligne jaune.

Les zones et les plages de couleurs mesurées sont enregistrées dans `hook_config.json`
(propre à votre écran, ignoré par git) et rechargées automatiquement.

Pour vérifier le réglage, gardez la jauge à l'écran et lancez `python hook.py --calibrate`.
Le dossier `debug/` reçoit alors la capture, les masques vert et jaune, ainsi qu'une image
annotée (rectangle magenta = zone verte, trait rouge = indicateur).

## Utilisation

```bash
python hook.py                       # boucle infinie
python hook.py --cycles 10 -v        # 10 cycles, logs détaillés
python hook.py --invert              # si Q/D agissent dans le sens opposé
python hook.py --input pyautogui     # touches via PyAutoGUI au lieu des scancodes
```

Vous avez 3 secondes après le lancement pour donner le focus à la fenêtre cible.

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
