"""Optional physical measurements complement the independent image classifier."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProduceItem(ABC):
    item_id: str
    name: str
    batch_code: str
    weight_gram: float
    _quality_score: float | None = field(init=False, default=None, repr=False)

    @property
    def quality_score(self):
        return self._quality_score

    def _grade(self, score):
        self._quality_score = round(100 * score, 2)
        return "A" if score >= .80 else "B" if score >= .55 else "C"

    @abstractmethod
    def calculate_grade(self):
        """Return a rule grade when the required measurements exist."""

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")} | {
            "quality_score": self.quality_score}


@dataclass
class Fruit(ProduceItem):
    brix_sweetness: float
    color_hue_range: tuple
    acidity_percent: float

    def calculate_grade(self):
        if self.brix_sweetness is None:
            return None
        spread = self.color_hue_range[1] - self.color_hue_range[0]
        return self._grade(.5 * min(self.brix_sweetness / 14, 1) + .5 * (1 - spread))

    def get_sugar_acid_ratio(self):
        if self.brix_sweetness is not None and self.acidity_percent:
            return round(self.brix_sweetness / self.acidity_percent, 2)


@dataclass
class Vegetable(ProduceItem):
    firmness_index: float
    foliage_health_score: float

    def calculate_grade(self):
        if self.firmness_index is not None and self.foliage_health_score is not None:
            return self._grade(.6 * self.firmness_index + .4 * self.foliage_health_score)

    def check_crispness_level(self):
        if self.firmness_index is not None:
            return "crisp" if self.firmness_index >= .7 else "soft"
