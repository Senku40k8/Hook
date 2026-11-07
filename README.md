# Hook

Boucle de test d'interface automatisée en Python (PyAutoGUI + OpenCV) :

1. appui sur `F`, puis latence aléatoire de 0.5 s à 1.5 s ;
2. attente de la bannière « Fish on the hook! Press the button to reel it! », puis nouvel appui sur `F` ;
3. suivi de la jauge : la zone verte se déplace à gauche et à droite, et la ligne jaune est
   maintenue sur elle par des à-coups sur `Q` (gauche) et `D` (droite), jusqu'à ce que la
   jauge disparaisse ;
4. appui sur `F` une fois la jauge disparue (le résultat est lu s'il est reconnaissable) ;
5. pause d'au moins 3 s, puis retour à l'étape 1.

La ligne jaune vise le centre de la zone verte, avec une anticipation dans le sens où la zone
se déplace. Les appuis sont d'autant plus longs que l'écart est grand. Seules les formes
jaunes fines et verticales sont prises pour la ligne : une icône jaune à côté de la jauge ne
fausse pas la mesure.

**Réglage automatique** : après chaque appui sur Q ou D, le bot mesure de combien la ligne a
réellement bougé. Il en déduit la vitesse de la ligne et dose ensuite ses appuis en
conséquence. Cette vitesse est mémorisée d'une partie à l'autre. Si la ligne part
systématiquement à l'opposé de l'appui, Q et D sont inversés automatiquement.

## Comportement humain et sessions

Activé par défaut (`--robotic` pour le désactiver) :

- **temps de réaction réalistes** : la plupart entre 250 et 350 ms, avec de temps en temps une
  réaction nettement plus lente ;
- **fatigue** : les réactions ralentissent très légèrement au fil d'une session, et
  redeviennent normales après une pause ;
- **petites imperfections** : une correction de temps en temps un peu tardive, un appui
  parfois un peu trop long ;
- **pauses variées entre les cycles** : la plupart de 3 à 4,5 s, environ 1 sur 8 de 5 à
  10 s, et environ 1 sur 30 de 10 à 20 s ;
- **sessions** : 20 à 45 min de pêche, puis 5 à 15 min de pause (`--no-breaks` pour
  désactiver).

## Réapprovisionnement en appâts

Quand la bannière « Equip bait before fishing. » apparaît (`templates/no_bait.png`), le bot
rachète des appâts tout seul, puis reprend la pêche :

1. `Q`, puis clic sur **Quick Submit** ;
2. clic sur l'**onglet au cube**, puis **Quick Sell** (vend les prises) ;
3. `Échap`, puis `R` pour ouvrir la boutique ;
4. clic sur **Universal Bait** (celui à 5 coquillages), sur le **double chevron** (quantité
   maximale), puis sur **Purchase** ;
5. `Échap`.

Les boutons sont dans `templates/restock/`. Ils sont recadrés à l'intérieur du contour rose
de survol, donc ils sont reconnus que la souris soit dessus ou non. L'icône du cube est
comparée par ses contours : l'onglet est reconnu qu'il soit gris ou orange. Les captures
d'origine sont conservées dans `templates/originals/`.

Le tout se fait comme un humain :

- la souris suit une trajectoire courbe, accélère puis ralentit en approchant du bouton ;
- la durée du geste dépend de la distance et de la taille du bouton (loi de Fitts) ;
- le curseur dépasse parfois légèrement la cible et se corrige, et il tremble très légèrement ;
- le point cliqué varie autour du centre du bouton, avec un court survol avant le clic ;
- le bot prend un temps de réaction quand un bouton apparaît, et hésite de temps en temps.

Le bot attend que chaque bouton soit affiché et immobile, à la fin de l'animation, avant de
cliquer. Si un bouton n'apparaît pas, il ferme les menus avec `Échap` et s'arrête avec une
alerte. Il ne refait jamais deux réapprovisionnements d'affilée : si la bannière revient
juste après un achat (plus de coquillages ?), il s'arrête au lieu de boucler.

## Contrôle, sécurité et statistiques

- **F8** : pause et reprise. Les touches sont relâchées pendant la pause, et un nouveau cycle
  démarre à la reprise. F8 écourte aussi une pause de session.
- **Arrêt automatique avec alerte sonore** après 3 cycles ratés de suite (bannière ou jauge
  absente, échec, jauge trop longue). Le nombre se règle avec `--max-failures N`.
- **Messages bloquants** : placez des captures recadrées de ces messages (inventaire plein,
  plus d'appât…) dans `templates/stop/`, par exemple `templates/stop/inventaire_plein.png`.
  Le bot s'arrête, avec une alerte, dès que l'un d'eux apparaît.
- **Résultat de la prise** (facultatif) : `templates/result_success.png` et
  `templates/result_fail.png`, recadrés sur le texte du message. Sans ces images, une jauge
  terminée compte comme « terminé ».
- **Plafonds** : `--duration 2h` (ou `1h30`, `90m`) et `--max-fish 100`.
- **Statistiques** : chaque cycle est ajouté à `logs/catches.csv`, qui s'ouvre dans Excel. On
  y trouve le résultat, l'attente de la bannière, la durée de la jauge, le nombre d'appuis
  sur Q et D, et la part du temps passée dans le vert. Un résumé s'affiche à l'arrêt.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Aucun réglage préalable n'est nécessaire. Au **premier lancement**, cliquez dans la fenêtre
du jeu : son titre est mémorisé. Aux lancements suivants, le bot met lui-même le jeu au
premier plan (pas besoin d'Alt+Tab). La barre des tâches et le sélecteur Alt+Tab sont
ignorés pendant le choix. Le bot ne travaille ensuite que dans cette fenêtre :

- la bannière et la jauge ne sont cherchées qu'à l'intérieur (une image visible dans un autre
  programme ne peut pas déclencher le bot) ;
- si le jeu quitte le premier plan, le bot se met en pause et n'envoie aucune touche.

> **Si F n'a aucun effet en jeu** : lancez le terminal **en administrateur**. Windows bloque,
> sans aucune erreur, les touches envoyées à un jeu lancé en administrateur (fréquent avec
> les anti-triche) depuis un programme qui ne l'est pas. Le bot affiche un avertissement
> quand il détecte ce cas.

```bash
python hook.py                       # boucle infinie
python hook.py --duration 2h         # 2 heures, puis arrêt avec un bip
python hook.py --max-fish 100        # 100 poissons, puis arrêt
python hook.py --record -v          # enregistre le suivi annoté dans debug/track/
python hook.py --list-windows        # titres des fenêtres ouvertes
python hook.py --window "Nom du jeu" # cible la fenêtre par son titre (et le mémorise)
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

Ces tests ne nécessitent ni écran ni fenêtre cible. Ils vérifient :

- les détecteurs et le correcteur, sur des images synthétiques ;
- le comportement humain, les sessions et les statistiques ;
- l'apprentissage de la vitesse de la ligne et l'inversion automatique de Q et D ;
- un cycle complet contre un faux jeu, avec bannière, jauge pilotée par Q/D et message de
  réussite.
