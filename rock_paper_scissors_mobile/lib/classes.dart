enum DetectionClasses { rock, paper, scissors, nothing }

extension DetectionClassesExtension on DetectionClasses {
  String get label {
    switch (this) {
      case DetectionClasses.rock:
        return "Rock";
      case DetectionClasses.paper:
        return "Paper";
      case DetectionClasses.scissors:
        return "Scissors";
      case DetectionClasses.nothing:
        return "Nothing";
    }
  }
}
