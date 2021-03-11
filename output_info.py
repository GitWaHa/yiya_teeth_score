import os


class OutputInfo(object):
    def __init__(self) -> None:
        super().__init__()

    def printScore(self,
                   patient_name: str,
                   all_score: list,
                   debug_info: str,
                   head: str = "aabb"):
        pre_score = all_score[0]
        mid_score = all_score[1]
        after_score = all_score[2]
        print(head, "#", patient_name, end=" ")
        print("#",
              pre_score[0],
              pre_score[1],
              pre_score[2],
              pre_score[3],
              pre_score[4],
              pre_score[5],
              pre_score[6],
              end=" ")
        print("#",
              mid_score[0],
              mid_score[1],
              mid_score[2],
              mid_score[3],
              mid_score[4],
              mid_score[5],
              mid_score[6],
              end=" ")
        print("#",
              after_score[0],
              after_score[1],
              after_score[2],
              after_score[3],
              after_score[4],
              after_score[5],
              after_score[6],
              end=" ")
        print("#", debug_info)

    def printError(self, patient_name: str, debug_info: str):
        error_score = [(0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0),
                       (0, 0, 0, 0, 0, 0, 0)]
        self.printScore(patient_name, error_score, debug_info, head="error")
