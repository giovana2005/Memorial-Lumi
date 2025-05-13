import random 

def funcao_objetivo_cb(candidato):
    """Computa a função objetivo no problema das caixas binárias
    
    Args:
      candidato: uma lista contendo os valores das caixas binárias do problema
      
    """
    return sum(candidato)

def gene_cb():
    """Sorteia um valor para uma caixa no problema das caixas binárias"""
    valores_possiveis = [0, 1]
    gene = random.choice(valores_possiveis)
    return gene

def cria_candidato_cb(n):
    """Cria uma lista com n valores zero ou um.
    
    Args:
      n: inteiro que representa o número de caixas.
    
    """
    candidato = []
    for _ in range(n):
        gene = gene_cb()
        candidato.append(gene)
    return candidato