import pygame
from typing import Tuple, Optional

class UIComponent:
    def draw(self, screen: pygame.Surface):
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

class Button(UIComponent):
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font, 
                 bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int],
                 active_color: Optional[Tuple[int, int, int]] = None,
                 callback=None):
        self.rect = rect
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.text_color = text_color
        self.active_color = active_color or bg_color
        self.callback = callback
        self.is_active = False

    def draw(self, screen: pygame.Surface):
        color = self.active_color if self.is_active else self.bg_color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        
        surf = self.font.render(self.text, True, self.text_color)
        text_rect = surf.get_rect(center=self.rect.center)
        screen.blit(surf, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True
        return False

class Slider(UIComponent):
    def __init__(self, rect: pygame.Rect, min_val: float, max_val: float, initial_val: float, 
                 callback=None):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.callback = callback
        self.dragging = False

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(ratio * self.rect.width)
        handle_rect = pygame.Rect(handle_x - 5, self.rect.y - 5, 10, self.rect.height + 10)
        pygame.draw.rect(screen, (200, 200, 200), handle_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.inflate(10, 10).collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value(event.pos[0])
                return True
        return False

    def _update_value(self, mouse_x: int):
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = max(0.0, min(1.0, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        if self.callback:
            self.callback(self.value)

class Graph:
    def __init__(self, rect: pygame.Rect, title: str, data_key: str, color: Tuple[int, int, int], 
                 min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.rect = rect
        self.title = title
        self.data_key = data_key
        self.color = color
        self.min_val = min_val
        self.max_val = max_val

    def draw(self, screen: pygame.Surface, data: list, font: pygame.font.Font):
        # Draw Title
        t_surf = font.render(self.title, True, self.color)
        screen.blit(t_surf, (self.rect.x, self.rect.y - 20))
        
        # Draw Box
        pygame.draw.rect(screen, (30, 30, 30), self.rect)
        pygame.draw.rect(screen, (80, 80, 80), self.rect, 1)
        
        if len(data) > 1:
            vals = data
            curr_min = min(vals) if self.min_val is None else self.min_val
            curr_max = max(vals) if self.max_val is None else self.max_val
            if curr_max <= curr_min: curr_max = curr_min + 1e-6
            
            points = []
            num_points = len(vals)

            for j, v in enumerate(vals):
                px = self.rect.x + (j / max(1, num_points - 1)) * self.rect.width
                norm_v = (v - curr_min) / (curr_max - curr_min)
                norm_v = max(0.0, min(1.0, norm_v))
                py = (self.rect.y + self.rect.height) - (norm_v * self.rect.height)
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(screen, self.color, False, points, 2)
            
            # Current Value
            curr = vals[-1]
            v_surf = font.render(f"{curr:.2f}", True, self.color)
            screen.blit(v_surf, (self.rect.right - v_surf.get_width(), self.rect.y - 20))

