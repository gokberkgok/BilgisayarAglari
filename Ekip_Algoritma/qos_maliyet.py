"""
QoS Maliyet Hesaplama Modülü

Bu modül, tüm algoritmalar ve GUI tarafından kullanılan 
ortak QoS maliyet hesaplama fonksiyonlarını içerir.

Amaç:
- Tek bir matematiksel gerçeklik sağlamak
- GUI ve algoritmaların aynı maliyet fonksiyonunu kullanmasını garanti etmek
- Bilimsel tutarlılık ve savunulabilirlik

Maliyet Bileşenleri:
1. Gecikme (Delay): Link delay + Processing delay
2. Güvenilirlik (Reliability): -log(reliability) ile maliyet
3. Kaynak (Resource): 1000/bandwidth ile maliyet
"""

import math
import networkx as nx


def compute_edge_cost(G, u, v, weights=None):
    """
    Bir edge'in QoS tabanlı maliyetini hesaplar.
    
    Args:
        G: NetworkX graph nesnesi
        u: Başlangıç düğümü
        v: Bitiş düğümü
        weights: dict - {'delay': w1, 'reliability': w2, 'resource': w3}
                 None ise eşit ağırlık kullanılır
    
    Returns:
        float: Edge'in toplam QoS maliyeti
    """
    if weights is None:
        weights = {'delay': 1.0, 'reliability': 1.0, 'resource': 1.0}
    
    edge_data = G[u][v]
    
    # 1. Gecikme maliyeti
    delay_cost = edge_data.get('link_delay', 0)
    
    # 2. Güvenilirlik maliyeti: -log(reliability)
    link_rel = edge_data.get('link_rel', 1.0)
    if link_rel > 0:
        reliability_cost = -math.log(link_rel)
    else:
        reliability_cost = float('inf')
    
    # 3. Kaynak maliyeti: 1000/bandwidth
    bandwidth = edge_data.get('bandwidth', 1)
    if bandwidth > 0:
        resource_cost = 1000.0 / bandwidth
    else:
        resource_cost = float('inf')
    
    # Toplam maliyet
    total = (weights['delay'] * delay_cost +
             weights['reliability'] * reliability_cost +
             weights['resource'] * resource_cost)
    
    return total


def compute_path_cost(G, path, weights=None):
    """
    Bir yolun toplam QoS maliyetini hesaplar.
    
    Args:
        G: NetworkX graph nesnesi
        path: list - Düğüm listesi [s, ..., d]
        weights: dict - {'delay': w1, 'reliability': w2, 'resource': w3}
                 None ise eşit ağırlık kullanılır
    
    Returns:
        dict: {'total_cost': float, 
               'delay': float, 
               'reliability': float, 
               'resource': float}
    """
    if weights is None:
        weights = {'delay': 1.0, 'reliability': 1.0, 'resource': 1.0}
    
    if not path or len(path) < 2:
        return {'total_cost': float('inf'), 'delay': 0, 'reliability': 0, 'resource': 0}
    
    total_delay = 0
    total_reliability_cost = 0
    total_resource_cost = 0
    
    # Edge maliyetleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Link delay
        total_delay += G[u][v].get('link_delay', 0)
        
        # Link reliability
        link_rel = G[u][v].get('link_rel', 1.0)
        if link_rel > 0:
            total_reliability_cost += -math.log(link_rel)
        else:
            total_reliability_cost = float('inf')
        
        # Bandwidth (resource)
        bandwidth = G[u][v].get('bandwidth', 1)
        if bandwidth > 0:
            total_resource_cost += 1000.0 / bandwidth
        else:
            total_resource_cost = float('inf')
    
    # Node processing delay (ara düğümler)
    for node in path[1:-1]:
        if 'proc_delay' in G.nodes[node]:
            total_delay += G.nodes[node]['proc_delay']
    
    # Node reliability (tüm düğümler)
    for node in path:
        if 'node_rel' in G.nodes[node]:
            node_rel = G.nodes[node]['node_rel']
            if node_rel > 0:
                total_reliability_cost += -math.log(node_rel)
            else:
                total_reliability_cost = float('inf')
    
    # Ağırlıklı toplam maliyet
    total_cost = (weights['delay'] * total_delay +
                  weights['reliability'] * total_reliability_cost +
                  weights['resource'] * total_resource_cost)
    
    return {
        'total_cost': total_cost,
        'delay': total_delay,
        'reliability_cost': total_reliability_cost,
        'resource_cost': total_resource_cost
    }


def validate_path_bandwidth(G, path, min_bandwidth):
    """
    Yoldaki tüm edge'lerin minimum bandwidth kısıtını sağlayıp sağlamadığını kontrol eder.
    
    Args:
        G: NetworkX graph nesnesi
        path: list - Düğüm listesi
        min_bandwidth: float - Minimum gerekli bandwidth
    
    Returns:
        tuple: (bool: geçerli mi?, list: geçersiz edge'ler)
    """
    if not path or len(path) < 2:
        return True, []
    
    invalid_edges = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        bandwidth = G[u][v].get('bandwidth', 0)
        
        if bandwidth < min_bandwidth:
            invalid_edges.append((u, v, bandwidth))
    
    is_valid = len(invalid_edges) == 0
    return is_valid, invalid_edges


def compute_path_metrics(G, path):
    """
    Yolun gerçek metriklerini hesaplar (GUI'de gösterilmek üzere).
    
    Args:
        G: NetworkX graph nesnesi
        path: list - Düğüm listesi
    
    Returns:
        dict: {'delay': float (ms), 
               'reliability': float (0-1), 
               'resource_cost': float,
               'hop_count': int}
    """
    if not path or len(path) < 2:
        return {'delay': 0, 'reliability': 1.0, 'resource_cost': 0, 'hop_count': 0}
    
    total_delay = 0
    combined_reliability = 1.0
    total_resource_cost = 0
    
    # Edge metrikleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Link delay
        total_delay += G[u][v].get('link_delay', 0)
        
        # Link reliability (çarpım)
        link_rel = G[u][v].get('link_rel', 1.0)
        combined_reliability *= link_rel
        
        # Resource cost
        bandwidth = G[u][v].get('bandwidth', 1)
        if bandwidth > 0:
            total_resource_cost += 1000.0 / bandwidth
    
    # Node processing delay (ara düğümler)
    for node in path[1:-1]:
        if 'proc_delay' in G.nodes[node]:
            total_delay += G.nodes[node]['proc_delay']
    
    # Node reliability (çarpım)
    for node in path:
        if 'node_rel' in G.nodes[node]:
            node_rel = G.nodes[node]['node_rel']
            combined_reliability *= node_rel
    
    return {
        'delay': total_delay,
        'reliability': combined_reliability,
        'resource_cost': total_resource_cost,
        'hop_count': len(path) - 1
    }
