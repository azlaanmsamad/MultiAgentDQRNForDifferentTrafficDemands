from gym.spaces.space import Space


class FixedActionsSpace(Space):
  """
  FixedActionSpace assumes that there is an enumerable set 
  of all possible actions that can become possible in the environment,
  even if it would run infinitely long.
  """

  def getAllActions(self) -> dict:
      """
      @return the complete dict of all actions that might become
      possible in this actionspace.
      the key is the int action number,
      the value a human-readable action value.
      eg {0:"UP", 1:"DOWN"}.
      """  
      raise NotImplementedError
