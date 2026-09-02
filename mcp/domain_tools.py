"""PkkMind 内置的确定性领域工具。

这些 handler 只做本地信息解释与字段检查，不访问真实账单、账号或工单系统，
也不会产生退款、修改配置等外部副作用。
"""
from typing import Any, Dict, Optional

from mcp.tool_manager import Tool


_ERROR_CODE_CATALOG: Dict[str, Dict[str, Any]] = {
    "400": {
        "meaning": "请求参数或格式不符合接口要求",
        "checks": ["核对请求体字段和数据类型", "检查 Content-Type", "查看接口返回的字段级错误"],
    },
    "401": {
        "meaning": "身份认证失败或凭证已失效",
        "checks": ["检查 Authorization 请求头", "确认 Token 是否过期", "重新登录后获取新凭证"],
    },
    "403": {
        "meaning": "身份已识别，但当前账号没有访问权限",
        "checks": ["确认账号角色和资源权限", "检查接口或功能是否对当前套餐开放", "联系管理员核验授权"],
    },
    "404": {
        "meaning": "请求的接口、页面或资源不存在",
        "checks": ["核对 URL 和资源 ID", "确认环境与接口版本", "检查资源是否已删除或迁移"],
    },
    "408": {
        "meaning": "请求处理超时",
        "checks": ["检查网络稳定性", "缩小单次请求数据量", "确认服务端是否繁忙"],
    },
    "429": {
        "meaning": "请求频率或额度超过限制",
        "checks": ["降低请求频率", "增加指数退避重试", "检查账户额度和限流策略"],
    },
    "500": {
        "meaning": "服务端内部处理异常",
        "checks": ["记录请求时间和请求 ID", "检查服务日志", "确认近期发布或配置变更"],
    },
    "502": {
        "meaning": "网关从上游服务获得了无效响应",
        "checks": ["检查上游服务状态", "确认网关路由配置", "检查代理和负载均衡日志"],
    },
    "503": {
        "meaning": "服务暂时不可用或正在维护",
        "checks": ["稍后重试", "检查服务健康状态", "确认是否处于维护窗口"],
    },
    "504": {
        "meaning": "网关等待上游服务响应超时",
        "checks": ["检查上游服务延迟", "检查网络链路", "核对网关超时配置"],
    },
}


async def error_code_lookup_handler(
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """解释常见 HTTP 错误码；未知错误码只返回安全的通用排查方向。"""
    del context
    error_code = str(params.get("error_code", "")).strip().upper()
    detail = _ERROR_CODE_CATALOG.get(error_code)
    if detail is None:
        detail = {
            "meaning": "当前本地目录中没有该错误码的确定解释",
            "checks": ["核对完整错误信息", "记录发生时间和运行环境", "结合官方文档或服务日志进一步确认"],
        }
    return {"error_code": error_code, **detail, "source": "pkkmind_local_catalog"}


async def billing_field_check_handler(
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """检查账单核验所需字段，不查询或修改任何真实账单。"""
    del context
    entities = params.get("entities") or {}
    required_fields = {
        "order_id": list(entities.get("order_id") or []),
        "amount": list(entities.get("amount") or []),
        "date": list(entities.get("date") or []),
    }
    available_fields = [name for name, values in required_fields.items() if values]
    missing_fields = [name for name, values in required_fields.items() if not values]
    return {
        "complete": not missing_fields,
        "available_fields": available_fields,
        "missing_fields": missing_fields,
        "values": required_fields,
        "requires_manual_verification": True,
        "side_effects_performed": False,
    }


def create_domain_tools() -> list[Tool]:
    """创建带有显式 Agent 白名单的内置领域工具。"""
    return [
        Tool(
            name="error_code_lookup",
            description="查询常见 HTTP 错误码含义和低风险排查步骤",
            handler=error_code_lookup_handler,
            schema={
                "type": "object",
                "properties": {"error_code": {"type": "string"}},
                "required": ["error_code"],
            },
            allowed_callers=("technical",),
        ),
        Tool(
            name="billing_field_check",
            description="检查账单核验需要的结构化字段是否齐全",
            handler=billing_field_check_handler,
            schema={
                "type": "object",
                "properties": {"entities": {"type": "object"}},
                "required": ["entities"],
            },
            allowed_callers=("billing",),
        ),
    ]
