import math


class QoSMetrics:
    def __init__(self, delay: float, reliability: float, resource: float):
        self.delay = delay
        self.reliability = reliability
        self.resource = resource

    def as_tuple(self):
        return self.delay, self.reliability, self.resource

    def as_dict(self):
        return {
            "Delay (ms)": round(self.delay, 3),
            "Reliability Cost": round(self.reliability, 5),
            "Resource Cost": round(self.resource, 5),
        }

    def dominates(self, other: "QoSMetrics") -> bool:
        return (
            self.delay <= other.delay
            and self.reliability <= other.reliability
            and self.resource <= other.resource
            and self.as_tuple() != other.as_tuple()
        )
