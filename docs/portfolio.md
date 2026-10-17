# Portfolio metadata

## Suggested repository name

`adversarial-uq-pinns`

## GitHub description

Three-person course project combining adversarial learning and physics-informed neural networks for synthetic beam-response uncertainty, with original notebooks and team history preserved.

## Topics

`physics-informed-neural-networks`, `uncertainty-quantification`, `scientific-machine-learning`, `adversarial-learning`, `generative-models`, `tensorflow`, `computational-science`, `python`

## CV bullet

Collaborated on an adversarial physics-informed neural-network model for uncertain beam response, combining latent-variable sampling, automatic differentiation, and adversarial training across boundary- and material-uncertainty experiments.

## Website description

In collaboration with Akash Yadav and Taiwo A. Adebiyi, I explored adversarial uncertainty quantification with physics-informed neural networks for a synthetic beam-deflection problem. The experiments use generator, encoder, and discriminator networks to learn from uncertain observations while penalizing a differential-equation residual. This portfolio copy preserves the team's original history and notebooks and documents the distinction between generated predictive variability and validated uncertainty coverage.

## Publication commands

After reviewing the prepared local repository, commit the cleanup and create an independent public repository under Nafeezat's account. Do not initialize a new Git history, squash the original commit, or force-push upstream.

```bash
git remote -v
git config user.name "Nafeezat A Ajenifuja"
git config user.email "164524949+nafeezatajenifuja@users.noreply.github.com"
git add -A
git commit -m "Organize team UQPINN experiments for reproducibility"
gh repo create nafeezatajenifuja/adversarial-uq-pinns --public --source=. --remote=origin --push --description "Three-person adversarial physics-informed learning project for synthetic beam-response uncertainty; original notebooks, history, and team attribution preserved."
git remote -v
git log --oneline
```

The full clone was made with `git clone https://github.com/TaiwoAdebiyi23/UQPINNS-GANS-ELEC_576-.git adversarial-uq-pinns`, followed by `git remote rename origin upstream`. The new `origin` is the portfolio copy; `upstream` remains the original team project.
