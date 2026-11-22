# kgnugen

Doomstop is a productivity tool that helps you stay focused by notifying you when you start doomscrolling. It can also provide virtual presenters with feedback on audience engagement, by displaying an average “attention”-score of the audience at any given moment. After a presentation, Doomstop also provides the presenter with information on how the attention progressed over time. The program transcribes the presentation, and identifies which topics resulted in lower attention scores.

Doomstop utilizes Logitech’s LogiInsight camera and software. To classify attention we use information about the user’s gaze, their hand positions, as well as an object detector for detecting a cellphone. We trained a logistic regression model on these features to classify the attention of different users. The model runs locally and all users are anonymous.

**On a Mac?** The LogiInsight colour camera isn't listed by macOS: use [`mac/`](mac/README.md) (Mac only).
